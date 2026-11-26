import numpy as np
from generate_battery_data import ocv_from_soc, docv_dsoc


class ExtendedKalmanFilterSoC:
    """
    Industry-standard 1st-order Thevenin Extended Kalman Filter (EKF) for BMS SoC estimation.
    Uses nominal cell parameters (3.0 Ah, healthy resistance), showing real-world sensitivity
    when cell parameters drift due to aging (80% SoH).
    """

    def __init__(self, nominal_capacity_ah: float = 3.0, dt: float = 1.0):
        self.cap_as = nominal_capacity_ah * 3600.0
        self.dt = dt
        self.Q = np.diag([1e-7, 1e-6])  # Process noise covariance
        self.R = np.array([[2.5e-4]])   # Voltage measurement noise covariance

    def estimate_cycle(self, voltages: np.ndarray, currents: np.ndarray, temps: np.ndarray, init_soc: float = 0.90) -> np.ndarray:
        """Runs EKF across a full drive cycle with realistic initial SoC error (init_soc=0.90)."""
        n = len(voltages)
        soc_est = np.zeros(n, dtype=np.float32)

        x = np.array([init_soc, 0.0], dtype=np.float64)  # State: [SoC, V_rc]
        P = np.diag([0.02, 0.005])

        for k in range(n):
            i_k = float(currents[k])
            t_k = float(temps[k])
            v_meas = float(voltages[k])

            # Nominal physics model parameters
            temp_factor = np.exp(0.035 * (25.0 - t_k))
            soc_factor = 1.0 + 0.6 * np.exp(-12.0 * max(x[0], 0.01))
            r0 = 0.028 * temp_factor * soc_factor
            r1 = 0.018 * temp_factor
            c1 = 1800.0
            tau = r1 * c1
            alpha = np.exp(-self.dt / tau)

            # 1. Prediction step
            if k > 0:
                A = np.array([[1.0, 0.0], [0.0, alpha]])
                x[0] = np.clip(x[0] - (currents[k - 1] * self.dt) / self.cap_as, 0.0, 1.0)
                x[1] = alpha * x[1] + r1 * (1.0 - alpha) * currents[k - 1]
                P = A @ P @ A.T + self.Q

            # 2. Measurement update step
            v_pred = float(ocv_from_soc(np.array([x[0]]))[0]) - i_k * r0 - x[1]
            H = np.array([[docv_dsoc(x[0]), -1.0]])
            S = H @ P @ H.T + self.R
            K = (P @ H.T) / S[0, 0]

            innovation = v_meas - v_pred
            x = x + (K.flatten() * innovation)
            x[0] = np.clip(x[0], 0.0, 1.0)
            P = (np.eye(2) - K @ H) @ P

            soc_est[k] = x[0]

        return soc_est