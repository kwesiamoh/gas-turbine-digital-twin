"""
Physics-Informed Neural Network (PINN) architecture and directional constraints.

Architecture:
  Input → BatchNorm1d → Linear(128, SiLU) → Dropout
        → Linear(64,  SiLU) → Dropout
        → Linear(32,  SiLU)
        → CO head (Linear 32→1) + NOx head (Linear 32→1)

The shared trunk learns common turbine-state representations while the two
output heads predict CO and NOx independently.

Loss function:
  L_total = L_data + λ · L_physics
  L_data    : MSE between predictions and measured CO/NOx
  L_physics : soft sign constraints on local CO and NOx sensitivities

MC Dropout uncertainty:
  The model remains in evaluation mode so BatchNorm uses learned running
  statistics, while only Dropout layers are re-enabled for stochastic forward
  passes. Running N passes yields a predictive mean and epistemic uncertainty
  estimate. Interval calibration is performed later on held-out data.
"""

import torch
import torch.nn as nn

from src.feature_engineering import PHYSICS_FEATS
from src.preprocessing import RAW_SENSORS

ALL_FEATURES = RAW_SENSORS + PHYSICS_FEATS

# Indices used by the directional constraints in the scaled input tensor.
IDX_TIT = ALL_FEATURES.index("TIT")
IDX_COMPRESSION = ALL_FEATURES.index("compression")
IDX_HUMIDITY = ALL_FEATURES.index("humidity_abs")

LAMBDA_PHYSICS = 0.15
DROPOUT_RATE = 0.20
MC_SAMPLES = 100

# Relative weights inside each pollutant's directional prior. They weight sign
# violations only; they are not coefficients in a hand-written emission model.
CO_TIT_WEIGHT = 0.6
CO_COMPRESSION_WEIGHT = 0.3
CO_HUMIDITY_WEIGHT = 0.2
NOX_TIT_WEIGHT = 0.6
NOX_HUMIDITY_WEIGHT = 0.4


class SiLU(nn.Module):
    """Sigmoid Linear Unit: x * sigmoid(x)."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class SoftSensorPINN(nn.Module):
    """Shared trunk with independent CO and NOx output heads."""

    def __init__(self, n_features: int, dropout: float = DROPOUT_RATE):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 128), SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), SiLU(),
        )
        self.co_head = nn.Linear(32, 1)
        self.nox_head = nn.Linear(32, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.bn_input(x)
        h = self.trunk(x)
        co = self.co_head(h)
        nox = self.nox_head(h)
        return torch.cat([co, nox], dim=1)

    def predict_with_uncertainty(self, x: torch.Tensor,
                                 n_samples: int = MC_SAMPLES):
        """
        Monte Carlo Dropout inference with BatchNorm frozen.

        Returns:
          mean : (n, 2) predictive mean in scaled target space
          std  : (n, 2) MC-Dropout standard deviation in scaled target space
        """
        if n_samples < 2:
            raise ValueError(
                "n_samples must be >= 2 for a finite MC standard deviation"
            )

        # Freeze BatchNorm and every other module first, then reactivate only
        # Dropout. This makes single-sample inference safe and prevents test
        # data from changing BatchNorm running statistics.
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(n_samples)], dim=0)

        self.eval()
        return preds.mean(dim=0), preds.std(dim=0)


def physics_residual(x_batch: torch.Tensor,
                     y_pred: torch.Tensor) -> torch.Tensor:
    """
    Penalise violations of physically motivated local emission directions.

    CO priors:
      d(CO)/d(TIT)         <= 0
      d(CO)/d(compression) <= 0
      d(CO)/d(humidity)    >= 0

    NOx priors:
      d(NOx)/d(TIT)        >= 0
      d(NOx)/d(humidity)   <= 0

    These are deliberately *soft directional priors*, not mass/energy balance
    equations. They regularise local sensitivities without forcing either
    pollutant toward a hand-written algebraic target.

    ``x_batch`` must have ``requires_grad=True``. Set LAMBDA_PHYSICS = 0 to
    recover the plain multi-output MLP objective.
    """
    if not x_batch.requires_grad:
        raise ValueError("physics_residual requires x_batch.requires_grad=True")

    co_pred = y_pred[:, 0]
    nox_pred = y_pred[:, 1]

    grad_co = torch.autograd.grad(
        outputs=co_pred.sum(),
        inputs=x_batch,
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_nox = torch.autograd.grad(
        outputs=nox_pred.sum(),
        inputs=x_batch,
        create_graph=True,
        retain_graph=True,
    )[0]

    dco_dtit = grad_co[:, IDX_TIT]
    dco_dcomp = grad_co[:, IDX_COMPRESSION]
    dco_dhum = grad_co[:, IDX_HUMIDITY]

    dnox_dtit = grad_nox[:, IDX_TIT]
    dnox_dhum = grad_nox[:, IDX_HUMIDITY]

    # CO: positive TIT/compression derivatives violate the expected decrease;
    # a negative humidity derivative violates the expected increase.
    co_penalty = (
        CO_TIT_WEIGHT * torch.relu(dco_dtit).pow(2).mean()
        + CO_COMPRESSION_WEIGHT * torch.relu(dco_dcomp).pow(2).mean()
        + CO_HUMIDITY_WEIGHT * torch.relu(-dco_dhum).pow(2).mean()
    )

    # NOx: negative TIT derivative violates the expected increase; a positive
    # humidity derivative violates the expected suppression by humidity.
    nox_penalty = (
        NOX_TIT_WEIGHT * torch.relu(-dnox_dtit).pow(2).mean()
        + NOX_HUMIDITY_WEIGHT * torch.relu(dnox_dhum).pow(2).mean()
    )

    # Averaging keeps the overall regularisation scale close to the previous
    # CO-only formulation while giving both targets physics guidance.
    return 0.5 * (co_penalty + nox_penalty)
