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
from models import SoCLSTM, SoC1DCNN

np.random.seed(42)
torch.manual_seed(42)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-informed and multi-scale rolling window features per drive cycle."""
    df = df.copy()
    df["power_w"] = df["voltage_v"] * df["current_a"]

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
    return df


class BatterySequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, seq_len: int = 30):
        X_list, y_list = [], []
        for _, group in df.groupby("cycle_id", sort=True):
            X_vals = group[feature_cols].to_numpy(dtype=np.float32)
            y_vals = group["soc"].to_numpy(dtype=np.float32)
            for i in range(seq_len - 1, len(group)):
                X_list.append(X_vals[i - seq_len + 1 : i + 1])
                y_list.append(y_vals[i])
        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def extract_aligned_tabular(df: pd.DataFrame, seq_len: int = 30) -> pd.DataFrame:
    """Extracts rows aligned 1-to-1 with BatterySequenceDataset."""
    chunks = [group.iloc[seq_len - 1 :].copy() for _, group in df.groupby("cycle_id", sort=True)]
    return pd.concat(chunks, ignore_index=True)


def train_pytorch_model(model: nn.Module, train_ds: Dataset, test_ds: Dataset, name: str, epochs: int = 14, batch_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training {name} on {device}...")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=0.02)

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
    """Monte Carlo Dropout for predictive uncertainty (mean +/- 95% confidence interval)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()

    # Activate Dropout layers only during inference
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
    """Exports PyTorch model to ONNX and measures single-sample CPU latency via ONNX Runtime."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        dummy,
        path,
        input_names=["telemetry_sequence"],
        output_names=["predicted_soc"],
        dynamic_axes={"telemetry_sequence": {0: "batch_size"}, "predicted_soc": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    sample_np = dummy.numpy()
    for _ in range(30):  # Warmup
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
    df = engineer_features(raw_df)

    feature_cols = [
        "voltage_v", "current_a", "temp_c", "power_w",
        "v_roll_mean_15s", "i_roll_mean_15s", "v_roll_std_15s",
        "v_roll_mean_60s", "i_roll_mean_60s", "v_roll_std_60s",
        "v_roll_mean_180s", "i_roll_mean_180s", "v_roll_std_180s",
    ]

    seq_len = 30
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # 1. Run physics-based Extended Kalman Filter on raw (unscaled) test cycles
    ekf = ExtendedKalmanFilterSoC(nominal_capacity_ah=3.0, dt=1.0)
    ekf_preds_full = []
    for _, grp in test_df.groupby("cycle_id", sort=True):
        est = ekf.estimate_cycle(
            grp["voltage_v"].to_numpy(),
            grp["current_a"].to_numpy(),
            grp["temp_c"].to_numpy(),
            init_soc=0.90,
        )
        ekf_preds_full.append(est[seq_len - 1 :])
    pred_ekf = np.concatenate(ekf_preds_full)

    # Scale features for ML / Deep Learning
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    train_ds = BatterySequenceDataset(train_df, feature_cols, seq_len=seq_len)
    test_ds = BatterySequenceDataset(test_df, feature_cols, seq_len=seq_len)

    tab_train = extract_aligned_tabular(train_df, seq_len=seq_len)
    tab_test = extract_aligned_tabular(test_df, seq_len=seq_len)

    X_train_tab, y_train_tab = tab_train[feature_cols].to_numpy(), tab_train["soc"].to_numpy()
    X_test_tab, y_test_tab = tab_test[feature_cols].to_numpy(), tab_test["soc"].to_numpy()

    # 2. Ridge Regression Baseline
    ridge = Ridge(alpha=1.0).fit(X_train_tab, y_train_tab)
    pred_ridge = np.clip(ridge.predict(X_test_tab), 0.0, 1.0)

    # 3. HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=0.05, random_state=42)
    hgb.fit(X_train_tab, y_train_tab)
    pred_hgb = np.clip(hgb.predict(X_test_tab), 0.0, 1.0)

    # 4. PyTorch LSTM + MC Dropout Uncertainty
    lstm_model, pred_lstm = train_pytorch_model(
        SoCLSTM(input_dim=len(feature_cols)), train_ds, test_ds, "PyTorch LSTM", epochs=14
    )
    lstm_mc_mean, lstm_mc_std = predict_mc_dropout(lstm_model, test_ds, n_samples=25)

    # 5. PyTorch 1D-CNN (Edge-optimized)
    cnn_model, pred_cnn = train_pytorch_model(
        SoC1DCNN(input_dim=len(feature_cols)), train_ds, test_ds, "PyTorch 1D-CNN", epochs=14
    )

    # Export ONNX & Benchmark CPU Latency
    lstm_ms, lstm_hz, lstm_kb = export_and_benchmark_onnx(
        lstm_model, seq_len, len(feature_cols), "models/soc_lstm.onnx"
    )
    cnn_ms, cnn_hz, cnn_kb = export_and_benchmark_onnx(
        cnn_model, seq_len, len(feature_cols), "models/soc_1dcnn.onnx"
    )

    # Metrics Table: Overall + Healthy (100% SoH) + Aged Stress Test (80% SoH)
    healthy_mask = tab_test["soh"].to_numpy() >= 0.99
    aged_mask = tab_test["soh"].to_numpy() < 0.90

    rows = []
    for name, preds in [
        ("Extended Kalman Filter (Physics)", pred_ekf),
        ("Ridge Regression (Linear)", pred_ridge),
        ("HistGradientBoosting (Tabular)", pred_hgb),
        ("PyTorch 1D-CNN (Edge DL)", pred_cnn),
        ("PyTorch LSTM (Sequential DL)", pred_lstm),
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

    print("\n=== ONNX RUNTIME CPU INFERENCE BENCHMARK (Batch Size = 1) ===")
    print(f"  PyTorch LSTM   -> {lstm_ms:.3f} ms/sample ({lstm_hz:,} Hz) | Size: {lstm_kb} KB")
    print(f"  PyTorch 1D-CNN -> {cnn_ms:.3f} ms/sample ({cnn_hz:,} Hz) | Size: {cnn_kb} KB")

    # Attach predictions for plotting
    tab_test["pred_ekf"] = pred_ekf
    tab_test["pred_hgb"] = pred_hgb
    tab_test["pred_cnn"] = pred_cnn
    tab_test["pred_lstm"] = pred_lstm
    tab_test["mc_mean"] = lstm_mc_mean
    tab_test["mc_std"] = lstm_mc_std

    # Plot 1: All 4 Test Cycles (25C, 0C, -10C Healthy + 10C Aged 80% SoH)
    test_cycles = sorted(tab_test["cycle_id"].unique())
    fig, axes = plt.subplots(len(test_cycles), 1, figsize=(11.5, 11.5))
    for ax, cid in zip(axes, test_cycles):
        sub = tab_test[tab_test["cycle_id"] == cid]
        t_min = sub["time_s"] / 60.0
        temp_c = sub["ambient_temp_c"].iloc[0]
        soh_pct = int(round(sub["soh"].iloc[0] * 100))
        prof = sub["profile"].iloc[0].capitalize()

        ax.plot(t_min, sub["soc"] * 100, label="Ground Truth", color="#111111", linewidth=2.2)
        ax.plot(t_min, sub["pred_ekf"] * 100, label="EKF (Physics)", linestyle=":", color="#ff7f0e", linewidth=1.8)
        ax.plot(t_min, sub["pred_cnn"] * 100, label="PyTorch 1D-CNN", linestyle="--", color="#1f77b4", linewidth=1.5)
        ax.plot(t_min, sub["pred_lstm"] * 100, label="PyTorch LSTM", color="#d62728", linewidth=1.6)
        ax.set_ylabel("SoC (%)")
        ax.set_title(f"Test Cycle #{cid} | Temp: {temp_c}°C | SoH: {soh_pct}% | Profile: {prof}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=8.5)
    axes[-1].set_xlabel("Time (minutes)")
    plt.tight_layout()
    plt.savefig("results/soc_benchmark_all_cycles.png", dpi=180)
    plt.close()

    # Plot 2: Uncertainty Quantification (MC Dropout 95% CI) on Cold (-10C) & Aged (80% SoH) Cycles
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, cid, title in zip(
        axes,
        [11, 12],
        ["Cold Weather Stress Test (-10°C, 100% SoH)", "Aged Battery Stress Test (+10°C, 80% SoH)"],
    ):
        sub = tab_test[tab_test["cycle_id"] == cid]
        t_min = sub["time_s"] / 60.0
        mean_pct = sub["mc_mean"] * 100
        ci95 = 1.96 * sub["mc_std"] * 100

        ax.plot(t_min, sub["soc"] * 100, label="Ground Truth SoC", color="#111111", linewidth=2.0)
        ax.plot(t_min, mean_pct, label="LSTM MC Mean", color="#d62728", linewidth=1.5)
        ax.fill_between(t_min, np.clip(mean_pct - ci95, 0, 100), np.clip(mean_pct + ci95, 0, 100),
                        color="#d62728", alpha=0.22, label="95% Confidence Band")
        ax.set_title(title)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("SoC (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig("results/soc_uncertainty_mc_dropout.png", dpi=180)
    plt.close()

    # Plot 3: Error Comparison Bar Chart (Healthy vs Aged 80% SoH)
    x = np.arange(len(res_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, res_df["Healthy MAE (%)"], width, label="Healthy Battery (100% SoH)", color="#1f77b4")
    ax.bar(x + width / 2, res_df["Aged 80% SoH MAE (%)"], width, label="Aged Battery Stress Test (80% SoH)", color="#d62728")
    ax.set_ylabel("Mean Absolute Error (% SoC)")
    ax.set_title("Model Robustness: Healthy vs. Degraded (80% SoH) Li-ion Cell")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(" (")[0] for m in res_df["Model"]], rotation=12, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/soc_robustness_healthy_vs_aged.png", dpi=180)
    plt.close()

    print("[OK] Saved 3 plots to results/")


if __name__ == "__main__":
    main()