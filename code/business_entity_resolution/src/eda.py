"""
EDA — quick exploratory analysis. Run standalone.

Usage:
    python src/eda.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, SEP


def main():
    print("=== Loading data (first 100K rows per source for speed) ===")
    s1 = pd.read_csv(TRAIN_S1, sep=SEP, dtype=str, nrows=100000, keep_default_na=False)
    s2 = pd.read_csv(TRAIN_S2, sep=SEP, dtype=str, nrows=100000, keep_default_na=False)
    s3 = pd.read_csv(TRAIN_S3, sep=SEP, dtype=str, nrows=100000, keep_default_na=False)
    gt = pd.read_csv(TRAIN_GT, sep=SEP, dtype=str, keep_default_na=False)

    print(f"\nShapes: S1={s1.shape}, S2={s2.shape}, S3={s3.shape}, GT={gt.shape}")

    # Column info
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"\n--- {name} ---")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Nulls/empty per column:")
        for col in df.columns:
            empty = (df[col].str.strip() == "").sum()
            print(f"    {col}: {empty} ({100*empty/len(df):.1f}%)")

    # Country distribution
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"\n  {name} country distribution:")
        print(df["country"].value_counts().to_string())

    # Ground truth stats
    print(f"\n--- Ground Truth ---")
    print(f"  Total rows: {len(gt)}")
    gt["n_matches"] = gt["matched_entity_ids"].apply(
        lambda x: len(x.split(",")) if x.strip() else 0
    )
    singletons = (gt["n_matches"] == 0).sum()
    print(f"  Singletons: {singletons} ({100*singletons/len(gt):.1f}%)")
    print(f"  With matches: {len(gt)-singletons}")
    matched = gt[gt["n_matches"] > 0]
    if len(matched) > 0:
        print(f"  Matches per entity: min={matched['n_matches'].min()}, "
              f"max={matched['n_matches'].max()}, "
              f"mean={matched['n_matches'].mean():.2f}, "
              f"median={matched['n_matches'].median():.0f}")

    # Name/address length stats
    print(f"\n--- Name/Address lengths (S1 sample) ---")
    s1["name_len"] = s1["business_name"].str.len()
    s1["addr_len"] = s1["business_address"].str.len()
    print(f"  Name:  mean={s1['name_len'].mean():.0f}, "
          f"median={s1['name_len'].median():.0f}, "
          f"max={s1['name_len'].max()}")
    print(f"  Addr:  mean={s1['addr_len'].mean():.0f}, "
          f"median={s1['addr_len'].median():.0f}, "
          f"max={s1['addr_len'].max()}")

    # Sample records
    print(f"\n--- Sample S1 records ---")
    print(s1.head(5).to_string())
    print(f"\n--- Sample S2 records ---")
    print(s2.head(5).to_string())
    print(f"\n--- Sample S3 records ---")
    print(s3.head(5).to_string())
    print(f"\n--- Sample GT records ---")
    print(gt.head(10).to_string())


if __name__ == "__main__":
    main()
