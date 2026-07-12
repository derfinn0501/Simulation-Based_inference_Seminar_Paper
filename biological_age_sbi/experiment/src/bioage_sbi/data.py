"""Biological-age data loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BASELINE_DATA_FILENAME, DEFAULT_FEATURE_SET_NAME, FEATURE_SETS, RAW_COLUMN_MAP


def find_mendeley_data_path(start: Path | None = None) -> Path:
    """Find the local Mendeley workbook from common notebook/root locations."""

    start = Path.cwd() if start is None else Path(start)
    candidates = [
        start / "biological_age_sbi/experiment/data/raw/mendeley_data.xlsx",
        start / "../data/raw/mendeley_data.xlsx",
        start / "data/raw/mendeley_data.xlsx",
    ]

    for parent in [start, *start.parents]:
        candidates.append(parent / "biological_age_sbi/experiment/data/raw/mendeley_data.xlsx")
        candidates.append(parent / "data/raw/mendeley_data.xlsx")

    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError("Could not find biological_age_sbi/experiment/data/raw/mendeley_data.xlsx")


def load_mendeley_data(path: Path | None = None, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """Load the Mendeley workbook."""

    path = find_mendeley_data_path() if path is None else Path(path)
    return pd.read_excel(path, sheet_name=sheet_name, na_values=["NA", "", " "])


def find_baseline_data_path(start: Path | None = None) -> Path:
    """Find the processed harmonized baseline dataset."""

    start = Path.cwd() if start is None else Path(start)
    candidates = [
        start / f"biological_age_sbi/experiment/data/processed/{BASELINE_DATA_FILENAME}",
        start / f"../data/processed/{BASELINE_DATA_FILENAME}",
        start / f"data/processed/{BASELINE_DATA_FILENAME}",
    ]

    for parent in [start, *start.parents]:
        candidates.append(parent / f"biological_age_sbi/experiment/data/processed/{BASELINE_DATA_FILENAME}")
        candidates.append(parent / f"data/processed/{BASELINE_DATA_FILENAME}")

    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"Could not find processed baseline dataset: {BASELINE_DATA_FILENAME}")


def load_baseline_data(path: Path | None = None) -> pd.DataFrame:
    """Load the harmonized baseline CSV used by the notebooks."""

    path = find_baseline_data_path() if path is None else Path(path)
    return pd.read_csv(path)


def yes_no_to_binary(values: pd.Series) -> pd.Series:
    """Convert numeric or yes/no columns to 0/1 floats."""

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return numeric.astype(float)

    normalized = values.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "yes": 1.0,
            "y": 1.0,
            "true": 1.0,
            "1": 1.0,
            "no": 0.0,
            "n": 0.0,
            "false": 0.0,
            "0": 0.0,
        }
    ).astype(float)


def prepare_model_frame(raw_df: pd.DataFrame, feature_set_name: str = DEFAULT_FEATURE_SET_NAME) -> pd.DataFrame:
    """Return a modeling frame with internal column names for one feature set."""

    if feature_set_name not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set: {feature_set_name!r}")

    feature_set = FEATURE_SETS[feature_set_name]
    model_columns = ["biological_age", *feature_set.continuous_columns, *feature_set.binary_columns]

    if all(col in raw_df.columns for col in model_columns):
        frame = raw_df[model_columns].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        return frame[model_columns].astype(float).reset_index(drop=True)

    missing = [RAW_COLUMN_MAP[col] for col in model_columns if RAW_COLUMN_MAP[col] not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing expected Mendeley columns: {missing}")

    frame = pd.DataFrame({"biological_age": pd.to_numeric(raw_df[RAW_COLUMN_MAP["biological_age"]], errors="coerce")})
    for col in feature_set.continuous_columns:
        frame[col] = pd.to_numeric(raw_df[RAW_COLUMN_MAP[col]], errors="coerce")
    for col in feature_set.binary_columns:
        frame[col] = yes_no_to_binary(raw_df[RAW_COLUMN_MAP[col]])

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame[model_columns].astype(float).reset_index(drop=True)


def split_train_holdout(
    frame: pd.DataFrame,
    holdout_fraction: float = 0.20,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic train/holdout split."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frame))
    holdout_size = int(round(len(frame) * holdout_fraction))
    holdout_idx = indices[:holdout_size]
    train_idx = indices[holdout_size:]

    train = frame.iloc[train_idx].sort_index().reset_index(drop=True)
    holdout = frame.iloc[holdout_idx].sort_index().reset_index(drop=True)
    return train, holdout
