"""
Data pipeline entry point.

Fetches the UCI Gas Turbine dataset, validates sensor readings, engineers
physics-derived features, and writes the processed dataset to disk.

Usage:
  python build_dataset.py

Outputs:
  data/raw/syngas_raw.csv            — unmodified UCI data
  data/processed/syngas_features.csv — validated data with physics features
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing     import (fetch_uci_dataset, validate_data,
                                   split_dataset, print_summary)
from src.feature_engineering import engineer_features, PHYSICS_FEATS


def build_dataset():
    print("=" * 65)
    print("  Data Ingestion & Feature Engineering")
    print("=" * 65)

    df_raw = fetch_uci_dataset()

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df_raw.to_csv("data/raw/syngas_raw.csv", index=False)
    print("  Raw data saved → data/raw/syngas_raw.csv")

    df_clean = validate_data(df_raw)
    df_full  = engineer_features(df_clean)
    train, test = split_dataset(df_full)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_full.to_csv("data/processed/syngas_features.csv", index=False)
    print("  Processed data saved → data/processed/syngas_features.csv")

    print_summary(df_full, train, test, PHYSICS_FEATS)

    print("\n→ Feed data/processed/syngas_features.csv into the model pipelines.\n")
    return df_full, train, test


if __name__ == "__main__":
    build_dataset()
