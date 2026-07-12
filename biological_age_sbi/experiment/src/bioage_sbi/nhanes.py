"""NHANES loading and KDM biological-age harmonization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


NHANES_2017_2020_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.XPT"

NHANES_2017_2020_FILES = {
    "P_DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR"],
    "P_BMX": ["SEQN", "BMXBMI"],
    "P_BPXO": ["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3"],
    "P_BIOPRO": ["SEQN", "LBXSCH", "LBXSTR", "LBXSCR", "LBXSBU"],
    "P_CBC": ["SEQN", "LBXPLTSI"],
    "P_HSCRP": ["SEQN", "LBXHSCRP"],
    "P_GHB": ["SEQN", "LBXGH"],
    "P_BPQ": ["SEQN", "BPQ020", "BPQ080"],
    "P_DIQ": ["SEQN", "DIQ010"],
    "P_MCQ": ["SEQN", "MCQ160A", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ220"],
    "P_SMQ": ["SEQN", "SMQ020"],
    "P_ALQ": ["SEQN", "ALQ111"],
}

KDM8_BIOMARKERS = ["tc", "tg", "hba1c", "bun", "creatinine", "sbp", "crp", "platelets"]


@dataclass(frozen=True)
class KDMFit:
    """Fitted KDM parameters for one calibration sample."""

    biomarker_params: pd.DataFrame
    s_r: float
    s_ba2: float
    s2: float
    nobs: int
    age_min: float
    age_max: float


def project_root(start: Path | None = None) -> Path:
    """Find the repository root from a script, notebook, or cwd."""

    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "biological_age_sbi/experiment/data").exists():
            return path
    raise FileNotFoundError("Could not find repository root containing biological_age_sbi/experiment/data")


def default_data_dirs(root: Path | None = None) -> tuple[Path, Path]:
    """Return the default raw NHANES directory and processed data directory."""

    root = project_root() if root is None else Path(root)
    data_dir = root / "biological_age_sbi/experiment/data"
    return data_dir / "raw/nhanes_2017_2020", data_dir / "processed"


def download_nhanes_2017_2020(raw_dir: Path, overwrite: bool = False) -> dict[str, str]:
    """Download selected official NHANES 2017-March 2020 XPT files."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for name in NHANES_2017_2020_FILES:
        url = NHANES_2017_2020_BASE_URL.format(name=name)
        path = raw_dir / f"{name}.XPT"
        if overwrite or not path.exists():
            urlretrieve(url, path)
        downloaded[name] = str(path)

    manifest = {
        "dataset": "NHANES 2017-March 2020 Pre-Pandemic",
        "source_base_url": NHANES_2017_2020_BASE_URL,
        "files": {
            name: {
                "url": NHANES_2017_2020_BASE_URL.format(name=name),
                "local_path": downloaded[name],
                "selected_columns": columns,
            }
            for name, columns in NHANES_2017_2020_FILES.items()
        },
        "notes": [
            "Files are official CDC NHANES SAS transport files.",
            "The harmonized KDM8 target is computed downstream from selected biomarkers.",
        ],
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return downloaded


def _read_xpt(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_sas(path, format="xport", encoding="utf-8")
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing expected columns in {path.name}: {missing}")
    return frame[columns].copy()


def load_nhanes_2017_2020(raw_dir: Path) -> pd.DataFrame:
    """Load and merge the selected NHANES files by respondent sequence number."""

    raw_dir = Path(raw_dir)
    merged: pd.DataFrame | None = None
    for name, columns in NHANES_2017_2020_FILES.items():
        frame = _read_xpt(raw_dir / f"{name}.XPT", columns)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="SEQN", how="left")
    if merged is None:
        raise ValueError("No NHANES files configured.")
    return merged


def _yes_no(series: pd.Series) -> pd.Series:
    return series.map({1.0: 1.0, 2.0: 0.0, 1: 1.0, 2: 0.0}).astype(float)


def _any_yes(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns]
    yes = values.eq(1).any(axis=1)
    no = values.eq(2).all(axis=1)
    return pd.Series(np.where(yes, 1.0, np.where(no, 0.0, np.nan)), index=frame.index)


def _log_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(values.where(values > 0.0))


def harmonize_nhanes_2017_2020(raw: pd.DataFrame, min_chronological_age: float = 45.0) -> pd.DataFrame:
    """Create a Mendeley-compatible NHANES modeling frame before KDM target calculation."""

    frame = pd.DataFrame(
        {
            "source_dataset": "nhanes_2017_2020",
            "source_id": raw["SEQN"].astype("Int64").astype("string"),
            "survey_years": "2017-March 2020",
            "bioage_method": "modified_kdm8_nhanes_2017_2020_biopro",
            "chronological_age": pd.to_numeric(raw["RIDAGEYR"], errors="coerce"),
            "gender": pd.to_numeric(raw["RIAGENDR"], errors="coerce"),
            "bmi": pd.to_numeric(raw["BMXBMI"], errors="coerce"),
            "sbp": raw[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1),
            "tc": _log_positive(raw["LBXSCH"]),
            "tg": _log_positive(raw["LBXSTR"]),
            "hba1c": _log_positive(raw["LBXGH"]),
            "bun": _log_positive(raw["LBXSBU"]),
            "creatinine": _log_positive(raw["LBXSCR"]),
            "crp": _log_positive(raw["LBXHSCRP"]),
            "platelets": _log_positive(raw["LBXPLTSI"]),
            "smoke": _yes_no(raw["SMQ020"]),
            "drink": _yes_no(raw["ALQ111"]),
            "hypertension": _yes_no(raw["BPQ020"]),
            "dyslipidemia": _yes_no(raw["BPQ080"]),
            "diabetes": _yes_no(raw["DIQ010"]),
            "cancer": _yes_no(raw["MCQ220"]),
            "arthritis": _yes_no(raw["MCQ160A"]),
            "cvd": _any_yes(raw, ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]),
            "koa": np.nan,
        }
    )
    frame = frame[frame["chronological_age"] >= min_chronological_age].reset_index(drop=True)
    return frame


def harmonize_mendeley_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """Create the same harmonized schema from the existing Mendeley CHARLS sheet."""

    arthritis = raw["Arthritis"].astype("string").str.strip().str.lower().map({"yes": 1.0, "no": 0.0})
    frame = pd.DataFrame(
        {
            "source_dataset": "mendeley_charls_fu_2025",
            "source_id": raw.index.astype("string"),
            "survey_years": raw["wave"].astype("string"),
            "bioage_method": "published_kdm8_charls_fu_2025",
            "chronological_age": np.nan,
            "gender": pd.to_numeric(raw["Gender"], errors="coerce"),
            "bmi": pd.to_numeric(raw["BMI"], errors="coerce"),
            "sbp": pd.to_numeric(raw["sbp.mean"], errors="coerce"),
            "tc": pd.to_numeric(raw["TC_mg.d L"], errors="coerce"),
            "tg": pd.to_numeric(raw["TG_mg.d L"], errors="coerce"),
            "hba1c": pd.to_numeric(raw["Hb A1c"], errors="coerce"),
            "bun": pd.to_numeric(raw["bun_mg.d L"], errors="coerce"),
            "creatinine": pd.to_numeric(raw["creatinine_mg.d L"], errors="coerce"),
            "crp": pd.to_numeric(raw["crp_mg.L"], errors="coerce"),
            "platelets": pd.to_numeric(raw["plt_10.9.L"], errors="coerce"),
            "smoke": pd.to_numeric(raw["Smoke"], errors="coerce"),
            "drink": pd.to_numeric(raw["Drink"], errors="coerce"),
            "hypertension": pd.to_numeric(raw["Hypertension"], errors="coerce"),
            "dyslipidemia": pd.to_numeric(raw["Dyslipidemia"], errors="coerce"),
            "diabetes": pd.to_numeric(raw["Diabetes"], errors="coerce"),
            "cancer": pd.to_numeric(raw["Cancer"], errors="coerce"),
            "arthritis": arthritis,
            "cvd": pd.to_numeric(raw["CVD"], errors="coerce"),
            "koa": pd.to_numeric(raw["KOA"], errors="coerce"),
            "biological_age": pd.to_numeric(raw["Biological Age"], errors="coerce"),
        }
    )
    return frame.dropna(subset=["biological_age"]).reset_index(drop=True)


def fit_kdm(frame: pd.DataFrame, biomarkers: list[str], age_col: str = "chronological_age") -> KDMFit:
    """Fit KDM biomarker-on-age parameters, following the BioAge kdm_calc formula."""

    fit_frame = frame[[age_col, *biomarkers]].dropna().astype(float)
    if len(fit_frame) < len(biomarkers) + 5:
        raise ValueError("Not enough complete rows to fit KDM.")

    age = fit_frame[age_col].to_numpy(dtype=float)
    rows = []
    for biomarker in biomarkers:
        y = fit_frame[biomarker].to_numpy(dtype=float)
        x = np.column_stack([np.ones_like(age), age])
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ coefficients
        residual = y - fitted
        residual_std = float(np.sqrt(np.sum(residual**2) / max(len(y) - 2, 1)))
        total_ss = float(np.sum((y - y.mean()) ** 2))
        residual_ss = float(np.sum(residual**2))
        r_squared = 0.0 if total_ss <= 0.0 else max(0.0, 1.0 - residual_ss / total_ss)
        rows.append(
            {
                "bm": biomarker,
                "q": float(coefficients[0]),
                "k": float(coefficients[1]),
                "s": max(residual_std, 1e-8),
                "r": r_squared,
            }
        )

    params = pd.DataFrame(rows)
    params["r1"] = np.abs((params["k"] / params["s"]) * np.sqrt(params["r"]))
    params["r2"] = np.abs(params["k"] / params["s"])
    params["n2"] = (params["k"] / params["s"]) ** 2

    rchar_denominator = float(params["r2"].sum())
    rchar = 0.0 if rchar_denominator <= 0.0 else float(params["r1"].sum() / rchar_denominator)
    age_min = float(np.nanmin(age))
    age_max = float(np.nanmax(age))
    s_r = ((1.0 - rchar**2) / max(rchar**2, 1e-8)) * (((age_max - age_min) ** 2) / (12.0 * len(params)))

    projected = _project_kdm_components(fit_frame, biomarkers, params, age_col)
    ba_ca = projected["ba_e"] - fit_frame[age_col].to_numpy(dtype=float)
    s2 = float(np.nanmean((ba_ca - np.nanmean(ba_ca)) ** 2))
    s_ba2 = max(s2 - s_r, 1e-6)

    return KDMFit(
        biomarker_params=params,
        s_r=float(s_r),
        s_ba2=float(s_ba2),
        s2=s2,
        nobs=int(len(fit_frame)),
        age_min=age_min,
        age_max=age_max,
    )


def _project_kdm_components(
    frame: pd.DataFrame,
    biomarkers: list[str],
    params: pd.DataFrame,
    age_col: str,
) -> dict[str, np.ndarray]:
    values = frame[biomarkers].astype(float)
    terms = []
    for biomarker in biomarkers:
        row = params.loc[params["bm"] == biomarker].iloc[0]
        terms.append((values[biomarker] - row["q"]) * (row["k"] / (row["s"] ** 2)))
    term_frame = pd.concat(terms, axis=1)
    n_missing = values.isna().sum(axis=1).to_numpy(dtype=float)
    n_observed = len(biomarkers) - n_missing
    ba_numerator = term_frame.sum(axis=1, skipna=True).to_numpy(dtype=float)
    ba_denominator = float(params["n2"].sum())
    ba_eo = ba_numerator / ba_denominator
    ba_e = (ba_eo / np.maximum(n_observed, 1.0)) * len(biomarkers)
    return {
        "n_missing": n_missing,
        "n_observed": n_observed,
        "ba_numerator": ba_numerator,
        "ba_denominator": np.full(len(frame), ba_denominator),
        "ba_e": ba_e,
        "age": frame[age_col].to_numpy(dtype=float),
    }


def project_kdm(
    frame: pd.DataFrame,
    biomarkers: list[str],
    fit: KDMFit,
    age_col: str = "chronological_age",
    max_missing: int = 2,
) -> pd.Series:
    """Project fitted KDM parameters onto a frame."""

    components = _project_kdm_components(frame, biomarkers, fit.biomarker_params, age_col)
    denominator = components["ba_denominator"] + (1.0 / fit.s_ba2)
    kdm = (components["ba_numerator"] + (components["age"] / fit.s_ba2)) / denominator
    kdm = np.where(components["n_missing"] > max_missing, np.nan, kdm)
    return pd.Series(kdm, index=frame.index, dtype=float)


def add_sex_specific_kdm(
    frame: pd.DataFrame,
    biomarkers: list[str] = KDM8_BIOMARKERS,
    sex_col: str = "gender",
    age_col: str = "chronological_age",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit and project KDM separately for NHANES male and female participants."""

    out = frame.copy()
    out["biological_age"] = np.nan
    out["biological_age_acceleration"] = np.nan
    fit_summary: dict[str, object] = {
        "algorithm": "Klemera-Doubal method via BioAge kdm_calc formula",
        "fit_scope": "sex-specific fit and projection within NHANES 2017-March 2020 complete KDM8 cases",
        "biomarkers": biomarkers,
        "sex_fits": {},
    }

    for sex_value, label in [(1.0, "male"), (2.0, "female")]:
        mask = out[sex_col].astype(float).eq(sex_value)
        sex_frame = out.loc[mask].copy()
        fit_frame = sex_frame.dropna(subset=[age_col, *biomarkers])
        fit = fit_kdm(fit_frame, biomarkers=biomarkers, age_col=age_col)
        projected = project_kdm(sex_frame, biomarkers=biomarkers, fit=fit, age_col=age_col)
        out.loc[mask, "biological_age"] = projected
        out.loc[mask, "biological_age_acceleration"] = projected - out.loc[mask, age_col]
        fit_summary["sex_fits"][label] = {
            "n_fit": fit.nobs,
            "age_min": fit.age_min,
            "age_max": fit.age_max,
            "s_r": fit.s_r,
            "s_ba2": fit.s_ba2,
            "s2": fit.s2,
            "biomarker_params": fit.biomarker_params.to_dict(orient="records"),
        }

    return out, fit_summary


def write_kdm_manifest(path: Path, fit_summary: dict[str, object], row_counts: dict[str, int]) -> None:
    """Write method metadata for the NHANES KDM8 processing step."""

    manifest = {
        "row_counts": row_counts,
        "method": fit_summary,
        "mendeley_source": {
            "dataset": "Raw Data of Biological Age",
            "doi": "10.17632/3rv7mf5pv9.1",
            "paper": "Fu et al. 2025, PLOS One, doi:10.1371/journal.pone.0335250",
            "reported_method": (
                "Klemera-Doubal biological age using total cholesterol, triglycerides, "
                "HbA1c, blood urea nitrogen, creatinine, systolic blood pressure, "
                "high-sensitivity CRP, and platelet count."
            ),
        },
        "nhanes_sources": {
            "dataset": "NHANES 2017-March 2020 Pre-Pandemic public use files",
            "files": NHANES_2017_2020_FILES,
            "base_url": NHANES_2017_2020_BASE_URL,
        },
        "notes": [
            "NHANES TC and TG use the Standard Biochemistry Profile variables LBXSCH and LBXSTR.",
            "Positive NHANES lab values are natural-log transformed to match the released Mendeley lab scale.",
            "Mendeley chronological age is not present in the released Sheet1 table, so it is left missing in the concat.",
            "The combined table keeps source_dataset and bioage_method because CHARLS KDM8 and NHANES KDM8 are comparable but not identical targets.",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2))
