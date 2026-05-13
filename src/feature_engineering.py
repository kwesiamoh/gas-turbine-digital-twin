"""
Physics-derived feature engineering for the gas turbine soft sensor.

Each engineered feature encodes a real thermodynamic relationship. These
features make the XGBoost model partially physics-informed even without
a PINN architecture, and they also serve as the PINN's named inputs.

Feature definitions:
  T_ratio      : TIT / TAT
                 Isentropic expansion efficiency proxy. Deviations from the
                 typical turbine ratio hint at fouling or incomplete combustion.

  compression  : CDP / AP (normalised to dimensionless via *1000 for mbar units)
                 Compressor pressure ratio. Higher compression → hotter combustion
                 → more complete oxidation → lower CO.

  T_drop       : TIT - TAT [°C]
                 Temperature drop across the turbine. Larger drop = more work
                 extracted. Correlates with CO burnout.

  humidity_abs : Derived from AH and AP using the Magnus approximation.
                 High absolute humidity dilutes O₂ → suppresses NOx, raises CO.
                 Formula: AH/100 * P_sat * 100 / (R_water * T_K)
                 where P_sat ≈ 6.1078 * exp(17.27*T_C / (T_C + 237.3)) mbar

  fouling_idx  : AFDP / AP (*1000 for mbar units)
                 Normalised filter pressure drop. Rising index = filter fouling
                 → less air intake → richer mixture → higher CO.

  specific_work: TEY / CDP [MWh / mbar]
                 Energy yield per unit of compression effort. Captures overall
                 combustion efficiency in a single ratio.

  dT_AT_TIT    : TIT - AT [°C]
                 Total temperature rise from ambient to turbine inlet.
                 Proxy for fuel energy input per combustion cycle.
"""

import numpy as np
import pandas as pd

PHYSICS_FEATS = [
    "T_ratio", "compression", "T_drop",
    "humidity_abs", "fouling_idx", "specific_work", "dT_AT_TIT",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-derived columns to df. Original sensor columns are preserved.

    Debug: if compression or fouling_idx values look unusually large/small,
    check that CDP and AFDP are in mbar (not Pa). The *1000 factor assumes
    both CDP and AP are in the same mbar unit.
    """
    out = df.copy()

    out["T_ratio"]     = df["TIT"] / df["TAT"]
    out["compression"] = df["CDP"] / df["AP"] * 1000   # both in mbar → dimensionless ratio

    out["T_drop"]      = df["TIT"] - df["TAT"]

    # Magnus approximation for saturation vapour pressure [mbar]
    T_C           = df["AT"]
    P_sat         = 6.1078 * np.exp(17.27 * T_C / (T_C + 237.3))
    out["humidity_abs"] = df["AH"] / 100.0 * P_sat * 100 / (461.5 * (T_C + 273.15))

    out["fouling_idx"]  = df["AFDP"] / df["AP"] * 1000  # mbar / mbar → dimensionless
    out["specific_work"]= df["TEY"] / df["CDP"]
    out["dT_AT_TIT"]    = df["TIT"] - df["AT"]

    return out
