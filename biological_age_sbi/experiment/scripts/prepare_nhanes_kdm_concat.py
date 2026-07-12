"""Download NHANES, compute KDM8 biological age, and concat with Mendeley."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = EXPERIMENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioage_sbi.nhanes import (  # noqa: E402
    KDM8_BIOMARKERS,
    add_sex_specific_kdm,
    default_data_dirs,
    download_nhanes_2017_2020,
    harmonize_mendeley_sheet,
    harmonize_nhanes_2017_2020,
    load_nhanes_2017_2020,
    write_kdm_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-raw", action="store_true", help="Re-download NHANES XPT files.")
    parser.add_argument(
        "--min-age",
        type=float,
        default=45.0,
        help="Minimum NHANES chronological age before KDM fitting/projection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir, processed_dir = default_data_dirs()
    processed_dir.mkdir(parents=True, exist_ok=True)

    download_nhanes_2017_2020(raw_dir=raw_dir, overwrite=args.overwrite_raw)
    nhanes_raw = load_nhanes_2017_2020(raw_dir=raw_dir)
    nhanes_harmonized = harmonize_nhanes_2017_2020(nhanes_raw, min_chronological_age=args.min_age)
    nhanes_with_ba, fit_summary = add_sex_specific_kdm(nhanes_harmonized, biomarkers=KDM8_BIOMARKERS)
    nhanes_complete = nhanes_with_ba.dropna(subset=["biological_age", "bmi", *KDM8_BIOMARKERS]).reset_index(drop=True)

    mendeley_path = raw_dir.parent / "mendeley_data.xlsx"
    mendeley_raw = pd.read_excel(mendeley_path, sheet_name="Sheet1", na_values=["NA", "", " "])
    mendeley_harmonized = harmonize_mendeley_sheet(mendeley_raw)

    combined = pd.concat([mendeley_harmonized, nhanes_complete], axis=0, ignore_index=True, sort=False)

    nhanes_path = processed_dir / "nhanes_2017_2020_kdm8_harmonized.csv"
    mendeley_path_out = processed_dir / "mendeley_kdm8_harmonized.csv"
    combined_path = processed_dir / "combined_mendeley_nhanes_kdm8.csv"
    manifest_path = processed_dir / "nhanes_2017_2020_kdm8_manifest.json"

    nhanes_complete.to_csv(nhanes_path, index=False)
    mendeley_harmonized.to_csv(mendeley_path_out, index=False)
    combined.to_csv(combined_path, index=False)

    row_counts = {
        "nhanes_raw_merged": int(len(nhanes_raw)),
        "nhanes_min_age": int(len(nhanes_harmonized)),
        "nhanes_kdm8_complete": int(len(nhanes_complete)),
        "mendeley_harmonized": int(len(mendeley_harmonized)),
        "combined": int(len(combined)),
    }
    write_kdm_manifest(manifest_path, fit_summary=fit_summary, row_counts=row_counts)

    print("Wrote:")
    print(f"  {nhanes_path} ({len(nhanes_complete)} rows)")
    print(f"  {mendeley_path_out} ({len(mendeley_harmonized)} rows)")
    print(f"  {combined_path} ({len(combined)} rows)")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
