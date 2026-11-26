import os
import numpy as np
import pandas as pd


def ocv_from_soc(soc: np.ndarray) -> np.ndarray:
    """Open Circuit Voltage (OCV) curve for a Li-ion NMC cell (3.0V - 4.2V)."""
    s = np.clip(soc, 0.0, 1.0)
    return (
        3.05
        + 1.15 * s
        - 0.25 * np.exp(-25.0 * s)
        + 0.12 * np.exp(4.0 * (s - 1.0))
        + 0.08 * np.sin(np.pi * s)
    )


def docv_dsoc(soc: float) -> float:
    """Analytical derivative d(OCV)/d(SoC) used by the Extended Kalman Filter."""
    s = float(np.clip(soc, 0.0, 1.0))
    return (
        1.15
        + 6.25 * np.exp(-25.0 * s)
        + 0.48 * np.exp(4.0 * (s - 1.0))
        + 0.08 * np.pi * np.cos(np.pi * s)
    )


def simulate_drive_cycle(
    cycle_id: int,
    ambient_temp_c: float,
    profile_type: str = "urban",
    soh: float = 1.0,
    dt: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates a Li-ion battery cell via a 1st-order Thevenin Equivalent Circuit Model.
    Supports State-of-Health (SoH) degradation: lower capacity and higher internal resistance.
    """
    rng = np.random.default_rng(seed + cycle_id)
    nominal_capacity_ah = 3.0
    actual_capacity_as = nominal_capacity_ah * soh * 3600.0
    resistance_aging_factor = 1.0 + 1.75 * (1.0 - soh)  # +35% resistance at 80% SoH

    max_steps = 4200
    t = np.arange(max_steps) * dt

    if profile_type == "urban":
        base = 2.2 + 1.8 * np.sin(2 * np.pi * t / 180.0) + 1.2 * np.sin(2 * np.pi * t / 47.0)
        bursts = pd.Series(rng.normal(0, 1.8, size=max_steps)).rolling(5, min_periods=1).mean().to_numpy()
        current_true = np.clip(base + bursts, -4.5, 11.0)
    elif profile_type == "highway":
        base = 4.6 + 1.2 * np.sin(2 * np.pi * t / 300.0)
        bursts = pd.Series(rng.normal(0, 1.0, size=max_steps)).rolling(8, min_periods=1).mean().to_numpy()
        current_true = np.clip(base + bursts, -2.5, 12.5)
    else:  # mixed dynamic
        base = 3.1 + 2.3 * np.cos(2 * np.pi * t / 240.0)
        bursts = pd.Series(rng.normal(0, 2.2, size=max_steps)).rolling(4, min_periods=1).mean().to_numpy()
        current_true = np.clip(base + bursts, -5.0, 13.0)

    soc = np.zeros(max_steps)
    voltage_meas = np.zeros(max_steps)
    current_meas = np.zeros(max_steps)
    temp = np.zeros(max_steps)

    soc[0] = rng.uniform(0.95, 1.0)
    temp[0] = ambient_temp_c
    v_rc = 0.0

    # Realistic current sensor bias on aged packs
    current_bias = rng.normal(0, 0.03) if soh < 0.92 else 0.0

    valid_steps = max_steps
    for k in range(max_steps):
        temp_factor = np.exp(0.035 * (25.0 - temp[k]))
        soc_factor = 1.0 + 0.6 * np.exp(-12.0 * max(soc[k], 0.01))
        r0 = 0.028 * temp_factor * soc_factor * resistance_aging_factor
        r1 = 0.018 * temp_factor * resistance_aging_factor
        c1 = 1800.0

        ocv = float(ocv_from_soc(np.array([soc[k]]))[0])
        v_term = ocv - current_true[k] * r0 - v_rc

        voltage_meas[k] = v_term + rng.normal(0, 0.005)
        current_meas[k] = current_true[k] + current_bias + rng.normal(0, 0.02)

        if voltage_meas[k] <= 2.85 or soc[k] <= 0.02:
            valid_steps = k + 1
            break

        if k < max_steps - 1:
            soc[k + 1] = np.clip(soc[k] - (current_true[k] * dt) / actual_capacity_as, 0.0, 1.0)
            tau = r1 * c1
            v_rc = v_rc * np.exp(-dt / tau) + r1 * (1.0 - np.exp(-dt / tau)) * current_true[k]
            heat_gen = (current_true[k] ** 2) * (r0 + r1) * 0.012
            cooling = 0.004 * (temp[k] - ambient_temp_c)
            temp[k + 1] = temp[k] + (heat_gen - cooling) * dt

    return pd.DataFrame({
        "cycle_id": cycle_id,
        "profile": profile_type,
        "ambient_temp_c": ambient_temp_c,
        "soh": soh,
        "time_s": t[:valid_steps],
        "voltage_v": np.round(voltage_meas[:valid_steps], 5),
        "current_a": np.round(current_meas[:valid_steps], 4),
        "temp_c": np.round(temp[:valid_steps], 3),
        "soc": np.round(soc[:valid_steps], 5),
    })


def main():
    os.makedirs("data", exist_ok=True)
    configs = [
        # Training cycles spanning temperatures (-10C to 25C) and aging (85% to 100% SoH)
        (1, 25.0, "urban", 1.00, "train"),
        (2, 25.0, "highway", 1.00, "train"),
        (3, 10.0, "urban", 1.00, "train"),
        (4, 10.0, "mixed", 0.90, "train"),
        (5, 0.0, "urban", 1.00, "train"),
        (6, 0.0, "highway", 0.90, "train"),
        (7, -10.0, "mixed", 1.00, "train"),
        (8, 25.0, "mixed", 0.90, "train"),
        (13, 25.0, "urban", 0.85, "train"),
        (14, 10.0, "highway", 0.85, "train"),
        (15, 0.0, "mixed", 0.85, "train"),
        (16, -10.0, "urban", 0.92, "train"),
        # Exact same 4 unseen test cycles (including 80% SoH extrapolation stress test)
        (9, 25.0, "mixed", 1.00, "test"),
        (10, 0.0, "mixed", 1.00, "test"),
        (11, -10.0, "urban", 1.00, "test"),
        (12, 10.0, "mixed", 0.80, "test"),
    ]

    dfs = []
    for cid, temp_c, prof, soh, split in configs:
        df_c = simulate_drive_cycle(cid, temp_c, prof, soh=soh, seed=200 + cid * 19)
        df_c["split"] = split
        dfs.append(df_c)

    full_df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join("data", "battery_drive_cycles.csv")
    full_df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(full_df):,} telemetry rows ({len(configs)} cycles, 85%-100% train + 80% SoH test) to {out_path}")


if __name__ == "__main__":
    main()