import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from ekf import ExtendedKalmanFilterSoC
from models import SoCLSTM, SoC1DCNN, HybridResidualTCN

np.random.seed(42)
torch.manual_seed(42)


def add_ekf_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Runs strictly causal EKF per cycle to extract physics SoC estimate and voltage innovation."""
    df = df.copy()
    ekf = ExtendedKalmanFilterSoC(nominal_capacity_ah=3.0, dt=1.0)
    ekf_soc_all = np.zeros(len(df), dtype=np.float32)
    ekf_innov_all = np.zeros(len(df), dtype=np.float32)

    for _, grp in df.groupby("cycle_id", sort=False):
        idx = grp.index.to_numpy()
        soc_est, innov = ekf.estimate_cycle(
            grp["voltage_v"].to_numpy(),
            grp["current_a"].to_numpy(),
            grp["temp_c"].to_numpy(),
            init_soc=0.90,
        )
        ekf_soc_all[idx] = soc_est
        ekf_innov_all[idx] = innov

    df["ekf_soc_raw"] = ekf_soc_all
    df["ekf_soc"] = ekf_soc_all
    df["ekf_innovation"] = ekf_innov_all
    df["ekf_residual"] = df["soc"] - df["ekf_soc_raw"]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-informed features: power, online impedance (dV/dI), and multi-scale rolling stats."""
    df = df.copy()
    df["power_w"] = df["voltage_v"] * df["current_a"]

    # Strictly causal online internal resistance estimate: R_est = -dV / dI during load transients
    dv = df.groupby("cycle_id")["voltage_v"].diff()
    di = df.groupby("cycle_id")["current_a"].diff()
    r_inst = np.where(np.abs(di) > 0.8, -dv / di, np.nan)
    r_inst = np.where((r_inst > 0.005) & (r_inst < 0.35), r_inst, np.nan)
    df["_r_inst"] = r_inst
    df["r_est_ohm"] = df.groupby("cycle_id")["_r_inst"].transform(
        lambda s: s.ffill().fillna(0.035).rolling(60, min_periods=1).median()
    )
    df.drop(columns=["_r_inst"], inplace=True)

    for win in [15, 60, 180]:
        df[f"v_roll_mean_{win}s"] = df.groupby("cycle_id")["voltage_v"].transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )
        df[f"i_roll_mean_{win}s"] = df.groupby("cycle_id")["current_a"].transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )
        df[f"v_roll_std_{win}s"] = df.groupby("cycle_id")["voltage_v"].transform(
            lambda s: s.rolling(win, min_periods=1).std().fillna(0.0)
        )
        df[f"innov_roll_mean_{win}s"] = df.groupby("cycle_id")["ekf_innovation"].transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )
    return df


class BatterySequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str = "soc", seq_len: int = 60):
        X_list, y_list = [], []
        for _, group in df.groupby("cycle_id", sort=True):
            X_vals = group[feature_cols].to_numpy(dtype=np.float32)
            y_vals = group[target_col].to_numpy(dtype=np.float32)
            for i in range(seq_len - 1, len(group)):
                X_list.append(X_vals[i - seq_len + 1 : i + 1])
                y_list.append(y_vals[i])
        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def extract_aligned_tabular(df: pd.DataFrame, seq_len: int = 60) -> pd.DataFrame:
    chunks = [group.iloc[seq_len - 1 :].copy() for _, group in df.groupby("cycle_id", sort=True)]
    return pd.concat(chunks, ignore_index=True)


def train_pytorch_model(
    model: nn.Module,
    train_ds: Dataset,
    test_ds: Dataset,
    name: str,
    epochs: int = 16,
    batch_size: int = 256,
    lr: float = 1.5e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training {name} on {device}...")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=0.015)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
        scheduler.step()
        if epoch in (1, epochs // 2, epochs):
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Huber Loss: {total_loss / len(train_ds):.6f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            preds.append(model(X_b.to(device)).cpu().numpy())

    return model, np.concatenate(preds)


def predict_mc_dropout(model: nn.Module, test_ds: Dataset, n_samples: int = 25, batch_size: int = 512):
    """Monte Carlo Dropout for predictive uncertainty (mean +/- 95% CI)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    mc_runs = []
    with torch.no_grad():
        for _ in range(n_samples):
            run_preds = [model(X_b.to(device)).cpu().numpy() for X_b, _ in loader]
            mc_runs.append(np.concatenate(run_preds))

    mc_runs = np.stack(mc_runs, axis=0)
    return mc_runs.mean(axis=0), mc_runs.std(axis=0)


def export_and_benchmark_onnx(model: nn.Module, seq_len: int, input_dim: int, path: str, n_iters: int = 500):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        dummy,
        path,
        input_names=["telemetry_sequence"],
        output_names=["predicted_output"],
        dynamic_axes={"telemetry_sequence": {0: "batch_size"}, "predicted_output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    sample_np = dummy.numpy()
    for _ in range(30):
        sess.run(None, {"telemetry_sequence": sample_np})

    t0 = time.perf_counter()
    for _ in range(n_iters):
        sess.run(None, {"telemetry_sequence": sample_np})
    elapsed_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
    freq_hz = 1000.0 / elapsed_ms
    size_kb = os.path.getsize(path) / 1024.0
    return round(elapsed_ms, 3), int(freq_hz), round(size_kb, 1)


def main():
    os.makedirs("results", exist_ok=True)
    raw_df = pd.read_csv("data/battery_drive_cycles.csv")
    df = add_ekf_signals(raw_df)
    df = engineer_features(df)

    base_feature_cols = [
        "voltage_v", "current_a", "temp_c", "power_w", "r_est_ohm",
        "v_roll_mean_15s", "i_roll_mean_15s", "v_roll_std_15s",
        "v_roll_mean_60s", "i_roll_mean_60s", "v_roll_std_60s",
        "v_roll_mean_180s", "i_roll_mean_180s", "v_roll_std_180s",
    ]
    hybrid_feature_cols = base_feature_cols + [
        "ekf_soc", "ekf_innovation",
        "innov_roll_mean_15s", "innov_roll_mean_60s", "innov_roll_mean_180s",
    ]

    seq_len = 60
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    scaler = StandardScaler()
    train_df[hybrid_feature_cols] = scaler.fit_transform(train_df[hybrid_feature_cols])
    test_df[hybrid_feature_cols] = scaler.transform(test_df[hybrid_feature_cols])

    tab_train = extract_aligned_tabular(train_df, seq_len=seq_len)
    tab_test = extract_aligned_tabular(test_df, seq_len=seq_len)

    X_train_tab, y_train_tab = tab_train[base_feature_cols].to_numpy(), tab_train["soc"].to_numpy()
    X_test_tab, y_test_tab = tab_test[base_feature_cols].to_numpy(), tab_test["soc"].to_numpy()
    pred_ekf = tab_test["ekf_soc_raw"].to_numpy()

    # 1. Ridge Regression Baseline
    ridge = Ridge(alpha=1.0).fit(X_train_tab, y_train_tab)
    pred_ridge = np.clip(ridge.predict(X_test_tab), 0.0, 1.0)

    # 2. HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=0.05, random_state=42)
    hgb.fit(X_train_tab, y_train_tab)
    pred_hgb = np.clip(hgb.predict(X_test_tab), 0.0, 1.0)

    # Datasets for Pure DL models
    train_ds_pure = BatterySequenceDataset(train_df, base_feature_cols, target_col="soc", seq_len=seq_len)
    test_ds_pure = BatterySequenceDataset(test_df, base_feature_cols, target_col="soc", seq_len=seq_len)

    # 3. PyTorch LSTM
    lstm_model, pred_lstm = train_pytorch_model(
        SoCLSTM(input_dim=len(base_feature_cols)), train_ds_pure, test_ds_pure, "PyTorch LSTM", epochs=16
    )

    # 4. PyTorch 1D-CNN
    cnn_model, pred_cnn = train_pytorch_model(
        SoC1DCNN(input_dim=len(base_feature_cols)), train_ds_pure, test_ds_pure, "PyTorch 1D-CNN", epochs=16
    )

    # 5. Hybrid EKF + Residual Dilated TCN (Physics-Informed AI)
    train_ds_hyb = BatterySequenceDataset(train_df, hybrid_feature_cols, target_col="ekf_residual", seq_len=seq_len)
    test_ds_hyb = BatterySequenceDataset(test_df, hybrid_feature_cols, target_col="ekf_residual", seq_len=seq_len)

    hybrid_model, pred_res = train_pytorch_model(
        HybridResidualTCN(input_dim=len(hybrid_feature_cols)),
        train_ds_hyb,
        test_ds_hyb,
        "Hybrid EKF + Residual TCN",
        epochs=18,
    )
    pred_hybrid = np.clip(pred_ekf + pred_res, 0.0, 1.0)

    # MC Dropout Uncertainty on the Hybrid Physics-AI model
    hyb_res_mean, hyb_res_std = predict_mc_dropout(hybrid_model, test_ds_hyb, n_samples=25)
    hyb_mc_mean = np.clip(pred_ekf + hyb_res_mean, 0.0, 1.0)
    hyb_mc_std = hyb_res_std

    # Export ONNX & Benchmark CPU Latency
    lstm_ms, lstm_hz, lstm_kb = export_and_benchmark_onnx(
        lstm_model, seq_len, len(base_feature_cols), "models/soc_lstm.onnx"
    )
    cnn_ms, cnn_hz, cnn_kb = export_and_benchmark_onnx(
        cnn_model, seq_len, len(base_feature_cols), "models/soc_1dcnn.onnx"
    )
    hyb_ms, hyb_hz, hyb_kb = export_and_benchmark_onnx(
        hybrid_model, seq_len, len(hybrid_feature_cols), "models/soc_hybrid_tcn.onnx"
    )

    # Benchmark Summary Table
    healthy_mask = tab_test["soh"].to_numpy() >= 0.99
    aged_mask = tab_test["soh"].to_numpy() < 0.90

    rows = []
    for name, preds in [
        ("Extended Kalman Filter (Physics)", pred_ekf),
        ("Ridge Regression (Linear)", pred_ridge),
        ("HistGradientBoosting (Tabular)", pred_hgb),
        ("PyTorch 1D-CNN (Edge DL)", pred_cnn),
        ("PyTorch LSTM (Sequential DL)", pred_lstm),
        ("Hybrid EKF + Residual TCN (Physics-AI)", pred_hybrid),
    ]:
        rows.append({
            "Model": name,
            "All MAE (%)": round(mean_absolute_error(y_test_tab, preds) * 100, 3),
            "Healthy MAE (%)": round(mean_absolute_error(y_test_tab[healthy_mask], preds[healthy_mask]) * 100, 3),
            "Aged 80% SoH MAE (%)": round(mean_absolute_error(y_test_tab[aged_mask], preds[aged_mask]) * 100, 3),
            "RMSE (%)": round(root_mean_squared_error(y_test_tab, preds) * 100, 3),
            "R2": round(r2_score(y_test_tab, preds), 5),
        })

    res_df = pd.DataFrame(rows)
    print("\n=== BENCHMARK RESULTS ON UNSEEN TEST CYCLES (INCLUDING 80% SoH AGED CELL) ===")
    print(res_df.to_string(index=False))
    res_df.to_csv("results/benchmark_metrics.csv", index=False)

    print("\n=== ONNX RUNTIME CPU INFERENCE BENCHMARK (Batch Size = 1, SeqLen = 60) ===")
    print(f"  PyTorch LSTM              -> {lstm_ms:.3f} ms/sample ({lstm_hz:,} Hz) | Size: {lstm_kb} KB")
    print(f"  PyTorch 1D-CNN            -> {cnn_ms:.3f} ms/sample ({cnn_hz:,} Hz) | Size: {cnn_kb} KB")
    print(f"  Hybrid EKF + Residual TCN -> {hyb_ms:.3f} ms/sample ({hyb_hz:,} Hz) | Size: {hyb_kb} KB")

    # Attach predictions for plotting
    tab_test["pred_ekf"] = pred_ekf
    tab_test["pred_lstm"] = pred_lstm
    tab_test["pred_hybrid"] = pred_hybrid
    tab_test["hyb_mc_mean"] = hyb_mc_mean
    tab_test["hyb_mc_std"] = hyb_mc_std

    # Plot 1: All 4 Test Cycles
    test_cycles = sorted(tab_test["cycle_id"].unique())
    fig, axes = plt.subplots(len(test_cycles), 1, figsize=(11.5, 11.5))
    for ax, cid in zip(axes, test_cycles):
        sub = tab_test[tab_test["cycle_id"] == cid]
        t_min = sub["time_s"] / 60.0
        temp_c = sub["ambient_temp_c"].iloc[0]
        soh_pct = int(round(sub["soh"].iloc[0] * 100))
        prof = sub["profile"].iloc[0].capitalize()

        ax.plot(t_min, sub["soc"] * 100, label="Ground Truth", color="#111111", linewidth=2.3)
        ax.plot(t_min, sub["pred_ekf"] * 100, label="EKF (Physics)", linestyle=":", color="#ff7f0e", linewidth=1.8)
        ax.plot(t_min, sub["pred_lstm"] * 100, label="PyTorch LSTM", linestyle="--", color="#1f77b4", linewidth=1.5)
        ax.plot(t_min, sub["pred_hybrid"] * 100, label="Hybrid EKF + Residual TCN", color="#2ca02c", linewidth=1.8)
        ax.set_ylabel("SoC (%)")
        ax.set_title(f"Test Cycle #{cid} | Temp: {temp_c}°C | SoH: {soh_pct}% | Profile: {prof}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=8.5)
    axes[-1].set_xlabel("Time (minutes)")
    plt.tight_layout()
    plt.savefig("results/soc_benchmark_all_cycles.png", dpi=180)
    plt.close()

    # Plot 2: Uncertainty Quantification (MC Dropout 95% CI on Hybrid Model)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, cid, title in zip(
        axes,
        [11, 12],
        ["Cold Weather Stress Test (-10°C, 100% SoH)", "Aged Battery Stress Test (+10°C, 80% SoH)"],
    ):
        sub = tab_test[tab_test["cycle_id"] == cid]
        t_min = sub["time_s"] / 60.0
        mean_pct = sub["hyb_mc_mean"] * 100
        ci95 = 1.96 * sub["hyb_mc_std"] * 100

        ax.plot(t_min, sub["soc"] * 100, label="Ground Truth SoC", color="#111111", linewidth=2.0)
        ax.plot(t_min, sub["pred_ekf"] * 100, label="Nominal EKF", linestyle=":", color="#ff7f0e", linewidth=1.6)
        ax.plot(t_min, mean_pct, label="Hybrid EKF+TCN Mean", color="#2ca02c", linewidth=1.6)
        ax.fill_between(
            t_min,
            np.clip(mean_pct - ci95, 0, 100),
            np.clip(mean_pct + ci95, 0, 100),
            color="#2ca02c",
            alpha=0.22,
            label="95% Confidence Band",
        )
        ax.set_title(title)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("SoC (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig("results/soc_uncertainty_mc_dropout.png", dpi=180)
    plt.close()

    # Plot 3: Healthy vs Aged 80% SoH Bar Chart
    x = np.arange(len(res_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, res_df["Healthy MAE (%)"], width, label="Healthy Battery (100% SoH)", color="#1f77b4")
    ax.bar(x + width / 2, res_df["Aged 80% SoH MAE (%)"], width, label="Aged Battery Stress Test (80% SoH)", color="#d62728")
    ax.set_ylabel("Mean Absolute Error (% SoC)")
    ax.set_title("Model Robustness: Healthy vs. Degraded (80% SoH) Li-ion Cell")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(" (")[0] for m in res_df["Model"]], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/soc_robustness_healthy_vs_aged.png", dpi=180)
    plt.close()

    print("[OK] Saved 3 updated plots to results/")


if __name__ == "__main__":
    main()