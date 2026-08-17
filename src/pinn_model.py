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
  The model remains in evaluation mode so BatchNorm uses its learned running
  statistics, while only Dropout layers are re-enabled for stochastic forward
  passes. Running N passes yields a predictive mean and epistemic uncertainty
  estimate.
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

    BatchNorm1d on input normalises across the batch dimension during training.
    MC Dropout inference keeps BatchNorm in evaluation mode and activates only
    Dropout layers, so single-sample inference is safe and running statistics
    are not modified by test data.

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

        Keeps BatchNorm in evaluation mode while enabling only Dropout layers,
        runs n_samples forward passes, and returns the empirical mean and std.

        Returns:
          mean : (n, 2) — predicted CO and NOx (scaled)
          std  : (n, 2) — uncertainty estimate per output
        """
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2 for a finite MC standard deviation")

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
    Soft directional constraint on the CO response.

    Expected local directions in scaled feature space:
      d(CO)/d(TIT)         <= 0
      d(CO)/d(compression) <= 0
      d(CO)/d(humidity)    >= 0

    The loss penalises only sign violations of these local derivatives rather
    than forcing CO toward an arbitrary linear equation. This makes the term a
    genuine directional prior while preserving the existing model architecture.

    ``x_batch`` must have ``requires_grad=True`` during training/validation.
    Set LAMBDA_PHYSICS = 0 to recover the plain data-driven MLP objective.
    """
    if not x_batch.requires_grad:
        raise ValueError("physics_residual requires x_batch.requires_grad=True")

    co_pred = y_pred[:, 0]
    grad_co = torch.autograd.grad(
        outputs=co_pred.sum(),
        inputs=x_batch,
        create_graph=True,
        retain_graph=True,
    )[0]

    dco_dtit = grad_co[:, IDX_TIT]
    dco_dcomp = grad_co[:, IDX_COMPRESSION]
    dco_dhum = grad_co[:, IDX_HUMIDITY]

    # relu(z) penalises z > 0 for features expected to reduce CO;
    # relu(-z) penalises z < 0 for humidity, which is expected to increase CO.
    penalty_tit = torch.relu(dco_dtit).pow(2).mean()
    penalty_comp = torch.relu(dco_dcomp).pow(2).mean()
    penalty_hum = torch.relu(-dco_dhum).pow(2).mean()

    return 0.6 * penalty_tit + 0.3 * penalty_comp + 0.2 * penalty_hum
