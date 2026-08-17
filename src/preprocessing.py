"""
Data ingestion, validation, and train/test splitting for the UCI Gas Turbine dataset.

UCI Gas Turbine CO and NOx Emission Dataset (ID: 551)
  36,733 hourly instances, Turkey gas turbine plant, 2011–2015
  License: CC BY 4.0
  DOI: https://doi.org/10.24432/C5WC95

Raw columns:
  AT [°C], AP [mbar], AH [%], AFDP [mbar], GTEP [mbar],
  TIT [°C], TAT [°C], CDP [mbar], TEY [MWh]  → inputs
  CO [mg/m³], NOx [mg/m³]                     → targets

Train/test split follows the original Kaya 2019 paper:
  Training: 2011–2013  |  Test: 2014–2015
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Physical operating bounds from the UCI dataset documentation.
# Used to detect and remove sensor faults.
BOUNDS = {
    "AT":   (-7,   38),
    "AP":   (985, 1037),
    "AH":   (24,  101),
    "AFDP": (2.0,  7.7),
    "GTEP": (17,   41),
    "TIT":  (1000, 1101),
    "TAT":  (511,  551),
    "CDP":  (9.8,  15.2),
    "TEY":  (100,  180),
    "CO":   (0.0,  44.2),
    "NOx":  (25.8, 120),
}

# Column name canonical mapping — ucimlrepo returns inconsistent capitalisation
# across versions (e.g. "NOX", "Nox", "co"). This maps all variants to a
# consistent form.
CANONICAL = {
    "at": "AT", "ap": "AP", "ah": "AH",
    "afdp": "AFDP", "gtep": "GTEP",
    "tit": "TIT", "tat": "TAT",
    "cdp": "CDP", "tey": "TEY",
    "co": "CO", "nox": "NOx", "year": "year",
}

RAW_SENSORS  = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP", "TEY"]
TARGETS      = ["CO", "NOx"]


def fetch_uci_dataset() -> pd.DataFrame:
    """
    Download UCI Gas Turbine dataset (ID=551) via ucimlrepo.
    Returns a single merged DataFrame with all rows.

    Debug: if CO/NOx columns are missing after fetch, the ucimlrepo version
    may be returning targets=None. The code handles this by checking the
    features DataFrame directly.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError("Run: pip install ucimlrepo")

    print("[1] Fetching UCI Gas Turbine dataset (ID=551)...")
    ds = fetch_ucirepo(id=551)
    X = ds.data.features.copy()

    # Some ucimlrepo versions expose CO/NOx in features, targets, or both.
    # Append only target columns that are not already present (case-insensitive)
    # so duplicate column names cannot silently propagate downstream.
    df = X.copy()
    if ds.data.targets is not None:
        y = ds.data.targets.copy()
        if isinstance(y, pd.Series):
            y = y.to_frame()
        existing = {str(col).lower() for col in df.columns}
        new_target_cols = [
            col for col in y.columns if str(col).lower() not in existing
        ]
        if new_target_cols:
            df = pd.concat([df, y[new_target_cols]], axis=1)

    # Normalise column names to canonical form
    rename_map = {
        col: CANONICAL[col.lower()]
        for col in df.columns
        if col.lower() in CANONICAL
    }
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        print(f"    Renamed columns: {rename_map}")

    missing = [c for c in ("CO", "NOx") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Target columns {missing} not found after normalisation.\n"
            f"Columns returned: {list(X.columns)}\n"
            f"Try: pip install --upgrade ucimlrepo"
        )

    # The temporal protocol depends on the explicit UCI ``year`` feature.
    # Never fabricate years from equal row blocks: the annual files have
    # different sizes, so doing so can silently corrupt the train/test split.
    if "year" not in df.columns:
        raise ValueError(
            "Required 'year' feature not returned by ucimlrepo. "
            "Upgrade ucimlrepo rather than reconstructing years from row order."
        )

    print(f"    Loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with physically impossible sensor values (sensor faults).

    Any row where at least one column falls outside the documented operating
    bounds is dropped. Reports per-column violation counts before dropping.
    """
    print("[2] Validating sensor readings against physical bounds...")

    flag_mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            out = ~df[col].between(lo, hi)
            if out.sum() > 0:
                print(f"    {col}: {out.sum()} values outside [{lo}, {hi}] — flagged")
                flag_mask |= out

    n_flagged = flag_mask.sum()
    df_clean  = df[~flag_mask].copy().reset_index(drop=True)
    print(f"    {n_flagged} rows removed ({100*n_flagged/len(df):.2f}%), "
          f"{len(df_clean):,} rows retained")
    return df_clean


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split following the original paper protocol:
      Training: 2011–2013
      Test:     2014–2015

    Temporal split is used (not random) to prevent data leakage from
    future readings into training.
    """
    train = df[df["year"] <= 2013].copy().reset_index(drop=True)
    test  = df[df["year"] >= 2014].copy().reset_index(drop=True)
    return train, test


def print_summary(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                  physics_feats: list):
    """Print dataset statistics and top correlations with each target."""
    print("\n── Dataset Summary ──────────────────────────────────────────────")
    print(f"  Total rows (clean)  : {len(df):,}")
    print(f"  Training rows       : {len(train):,}  (2011–2013)")
    print(f"  Test rows           : {len(test):,}  (2014–2015)")
    print(f"  Missing values      : {df.isna().sum().sum()} (none expected)")

    print("\n── Target ranges ────────────────────────────────────────────────")
    for col in TARGETS:
        s = df[col]
        print(f"  {col:5s}  mean={s.mean():.2f}  std={s.std():.2f}"
              f"  [{s.min():.2f}, {s.max():.2f}]")

    all_cols = RAW_SENSORS + physics_feats
    for tgt in TARGETS:
        print(f"\n── Top correlations with {tgt} ─────────────────────────────────")
        corr = df[all_cols + [tgt]].corr()[tgt].drop(tgt)
        top5 = corr.abs().nlargest(5)
        for feat, val in top5.items():
            sign = "+" if corr[feat] > 0 else "-"
            print(f"  {sign}{abs(val):.3f}  {feat}")
