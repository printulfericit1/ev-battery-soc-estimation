import torch
import torch.nn as nn


class SoCLSTM(nn.Module):
    """2-layer LSTM with MC Dropout head for uncertainty-aware SoC estimation."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class SoC1DCNN(nn.Module):
    """Lightweight Temporal 1D-CNN optimized for low-latency embedded BMS inference."""

    def __init__(self, input_dim: int, dropout: float = 0.10):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        feat = self.conv_net(x)
        return self.head(feat).squeeze(-1)


class HybridResidualTCN(nn.Module):
    """
    Physics-Informed Residual Dilated 1D-CNN (TCN).
    Predicts the bounded correction delta_SoC = SoC_true - SoC_EKF in [-0.25, +0.25]
    using dilated convolutions (dilations 1, 2, 4, 8) to capture RC polarization history.
    """

    def __init__(self, input_dim: int, dropout: float = 0.10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, F, T)
        h = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        # Combine instantaneous last step + full-window average context
        pooled = torch.cat([h[:, :, -1], h.mean(dim=-1)], dim=1)
        return 0.25 * torch.tanh(self.head(pooled).squeeze(-1))