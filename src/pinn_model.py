"""
Physics-Informed Neural Network (PINN) architecture and physics constraint.

Architecture:
  Input → BatchNorm1d → Linear(128, SiLU) → Dropout
        → Linear(64,  SiLU) → Dropout
        → Linear(32,  SiLU)
        → CO head (Linear 32→1) + NOx head (Linear 32→1)

  Shared trunk with two independent output heads allows the model to learn
  correlated representations of CO and NOx simultaneously.

  SiLU (x * sigmoid(x)) is used instead of ReLU to avoid dying neurons
  in regression tasks and for smoother gradients near zero.

Loss function:
  L_total = L_data + λ · L_physics
  L_data    : MSE between predictions and measured CO/NOx
  L_physics : directional constraint residual (see physics_residual docstring)

MC Dropout uncertainty:
  Dropout is kept active at inference time. Running N forward passes
  yields a distribution; mean and std are used as prediction and uncertainty.
  If std values collapse to near zero: check DROPOUT_RATE >= 0.1 and that
  model is in .train() mode (not .eval()) during MC sampling.
"""

import torch
import torch.nn as nn

from src.feature_engineering import PHYSICS_FEATS
from src.preprocessing import RAW_SENSORS

ALL_FEATURES    = RAW_SENSORS + PHYSICS_FEATS

# Indices used by the physics constraint to access specific features
# from the scaled input tensor.
IDX_TIT         = ALL_FEATURES.index("TIT")
IDX_COMPRESSION = ALL_FEATURES.index("compression")
IDX_HUMIDITY    = ALL_FEATURES.index("humidity_abs")

LAMBDA_PHYSICS  = 0.15   # physics loss weight; set to 0 for plain MLP baseline
DROPOUT_RATE    = 0.20
MC_SAMPLES      = 100    # number of MC dropout forward passes at inference


class SiLU(nn.Module):
    """Sigmoid Linear Unit: x * sigmoid(x). Smoother than ReLU for regression."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SoftSensorPINN(nn.Module):
    """
    Shared trunk with independent CO and NOx output heads.

    BatchNorm1d on input normalises across the batch dimension.
    Note: BatchNorm requires batch_size >= 2. At single-sample inference,
    call model.eval() before predict_with_uncertainty to use running stats —
    but note this also disables dropout, so use batch inference for MC Dropout.

    Xavier uniform initialisation is used for all Linear layers; this is
    better conditioned for regression than PyTorch's default Kaiming init.
    """

    def __init__(self, n_features: int, dropout: float = DROPOUT_RATE):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 128), SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),         SiLU(), nn.Dropout(dropout),
            nn.Linear(64,  32),         SiLU(),
        )
        self.co_head  = nn.Linear(32, 1)
        self.nox_head = nn.Linear(32, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x   = self.bn_input(x)
        h   = self.trunk(x)
        co  = self.co_head(h)
        nox = self.nox_head(h)
        return torch.cat([co, nox], dim=1)   # shape: (batch, 2)

    def predict_with_uncertainty(self, x: torch.Tensor,
                                  n_samples: int = MC_SAMPLES):
        """
        Monte Carlo Dropout inference.

        Keeps dropout active via model.train(), runs n_samples forward passes,
        and returns the empirical mean and std across passes.

        Returns:
          mean : (n, 2) — predicted CO and NOx (scaled)
          std  : (n, 2) — uncertainty estimate per output
        """
        self.train()   # activates dropout for MC sampling
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        self.eval()
        return preds.mean(dim=0), preds.std(dim=0)


def physics_residual(x_batch: torch.Tensor,
                     y_pred: torch.Tensor) -> torch.Tensor:
    """
    Soft directional constraint: CO should decrease with higher TIT and
    compression, and increase with higher absolute humidity.

    Encoded as a linear approximation in scaled feature space:
      CO_eq ≈ -0.6 * TIT_norm - 0.3 * compression_norm + 0.2 * humidity_norm

    This is not a strict mass-balance equation — it is a learned directional
    prior that keeps predictions thermodynamically consistent. Set
    LAMBDA_PHYSICS = 0 to disable and train a plain data-driven MLP.

    Debug: if L_physics dominates L_data in early training, reduce
    LAMBDA_PHYSICS or verify that feature scaling is consistent with the
    coefficients above (they assume RobustScaler output, roughly [-2, 2]).
    """
    TIT_norm  = x_batch[:, IDX_TIT]
    comp_norm = x_batch[:, IDX_COMPRESSION]
    hum_norm  = x_batch[:, IDX_HUMIDITY]

    co_eq           = -0.6 * TIT_norm - 0.3 * comp_norm + 0.2 * hum_norm
    co_pred_scaled  = y_pred[:, 0]   # column 0 is CO (scaled)
    return ((co_pred_scaled - co_eq) ** 2).mean()
