"""Shared configuration for the biological-age simulator."""

from __future__ import annotations

from dataclasses import dataclass

BIOAGE_COL = "Biological Age"
BASELINE_DATASET_NAME = "combined_mendeley_nhanes_kdm8"
BASELINE_DATA_FILENAME = "combined_mendeley_nhanes_kdm8.csv"
MODEL_NAME = "set_e_kdm8_common_lab"

RAW_COLUMN_MAP = {
    "biological_age": BIOAGE_COL,
    "bmi": "BMI",
    "sbp": "sbp.mean",
    "smoke": "Smoke",
    "drink": "Drink",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "dyslipidemia": "Dyslipidemia",
    "cancer": "Cancer",
    "cvd": "CVD",
    "arthritis": "Arthritis",
    "koa": "KOA",
    "platelets": "plt_10.9.L",
    "crp": "crp_mg.L",
    "hba1c": "Hb A1c",
    "creatinine": "creatinine_mg.d L",
    "bun": "bun_mg.d L",
    "tc": "TC_mg.d L",
    "tg": "TG_mg.d L",
}


@dataclass(frozen=True)
class FeatureSet:
    """Column specification for one biological-age observation design."""

    name: str
    description: str
    continuous_columns: tuple[str, ...]
    binary_columns: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.continuous_columns, *self.binary_columns)


FEATURE_SETS = {
    "set_a_easy_non_lab": FeatureSet(
        name="set_a_easy_non_lab",
        description="Easy non-lab indicators: BMI, systolic blood pressure, smoking, and drinking.",
        continuous_columns=("bmi", "sbp"),
        binary_columns=("smoke", "drink"),
    ),
    "set_b_non_lab_diagnosis": FeatureSet(
        name="set_b_non_lab_diagnosis",
        description=(
            "Current non-lab diagnosis-expanded set: BMI, systolic blood pressure, smoking, "
            "drinking, hypertension, diabetes, CVD, arthritis, and KOA."
        ),
        continuous_columns=("bmi", "sbp"),
        binary_columns=("smoke", "drink", "hypertension", "diabetes", "cvd", "arthritis", "koa"),
    ),
    "set_c_extended_non_lab_diagnosis": FeatureSet(
        name="set_c_extended_non_lab_diagnosis",
        description=(
            "Extended non-lab diagnosis set: Set B plus dyslipidemia and cancer."
        ),
        continuous_columns=("bmi", "sbp"),
        binary_columns=(
            "smoke",
            "drink",
            "hypertension",
            "dyslipidemia",
            "diabetes",
            "cancer",
            "cvd",
            "arthritis",
            "koa",
        ),
    ),
    "set_f_common_non_lab_diagnosis": FeatureSet(
        name="set_f_common_non_lab_diagnosis",
        description=(
            "Common non-lab diagnosis-expanded set for the combined Mendeley+NHANES data: "
            "BMI, systolic blood pressure, smoking, drinking, hypertension, "
            "dyslipidemia, diabetes, cancer, CVD, and arthritis. KOA is excluded "
            "because NHANES does not provide a comparable KOA variable."
        ),
        continuous_columns=("bmi", "sbp"),
        binary_columns=(
            "smoke",
            "drink",
            "hypertension",
            "dyslipidemia",
            "diabetes",
            "cancer",
            "cvd",
            "arthritis",
        ),
    ),
    "set_d_lab_enriched": FeatureSet(
        name="set_d_lab_enriched",
        description=(
            "Lab-enriched set: extended non-lab diagnoses plus platelet count, CRP, HbA1c, "
            "creatinine, BUN, total cholesterol, and triglycerides."
        ),
        continuous_columns=("bmi", "sbp", "platelets", "crp", "hba1c", "creatinine", "bun", "tc", "tg"),
        binary_columns=(
            "smoke",
            "drink",
            "hypertension",
            "dyslipidemia",
            "diabetes",
            "cancer",
            "cvd",
            "arthritis",
            "koa",
        ),
    ),
    "set_e_kdm8_common_lab": FeatureSet(
        name="set_e_kdm8_common_lab",
        description=(
            "KDM8 common lab-rich baseline for the combined Mendeley+NHANES data: BMI, "
            "systolic blood pressure, platelet count, CRP, HbA1c, creatinine, BUN, "
            "total cholesterol, triglycerides, smoking, drinking, hypertension, "
            "dyslipidemia, diabetes, cancer, CVD, and arthritis. KOA is excluded "
            "because NHANES does not provide a comparable KOA variable."
        ),
        continuous_columns=("bmi", "sbp", "platelets", "crp", "hba1c", "creatinine", "bun", "tc", "tg"),
        binary_columns=(
            "smoke",
            "drink",
            "hypertension",
            "dyslipidemia",
            "diabetes",
            "cancer",
            "cvd",
            "arthritis",
        ),
    ),
}

DEFAULT_FEATURE_SET_NAME = MODEL_NAME

CONTINUOUS_MODEL_COLUMNS = list(FEATURE_SETS[MODEL_NAME].continuous_columns)
BINARY_MODEL_COLUMNS = list(FEATURE_SETS[MODEL_NAME].binary_columns)
LATENT_MODEL_COLUMNS = ["metabolic_risk", "cardiovascular_risk", "joint_burden", "behavior_risk"]
MODEL_COLUMNS = ["biological_age", *CONTINUOUS_MODEL_COLUMNS, *BINARY_MODEL_COLUMNS]

OBSERVED_KEY_BY_COLUMN = {col: f"observed_{col}" for col in RAW_COLUMN_MAP if col != "biological_age"}
TRUE_KEY_BY_COLUMN = {col: f"true_{col}" for col in RAW_COLUMN_MAP if col != "biological_age"}

CONDITION_KEYS = [OBSERVED_KEY_BY_COLUMN[col] for col in [*CONTINUOUS_MODEL_COLUMNS, *BINARY_MODEL_COLUMNS]]
TRUE_KEYS = [TRUE_KEY_BY_COLUMN[col] for col in [*CONTINUOUS_MODEL_COLUMNS, *BINARY_MODEL_COLUMNS]]
LATENT_KEYS = [f"latent_{col}" for col in LATENT_MODEL_COLUMNS]

CONTINUOUS_FEATURES = {
    "bmi": ["intercept", "age_z", "age_z2"],
    "sbp": ["intercept", "age_z", "age_z2", "bmi_z"],
}

BINARY_FEATURES = {
    "smoke": ["intercept", "age_z", "age_z2"],
    "drink": ["intercept", "age_z", "age_z2", "smoke"],
    "hypertension": ["intercept", "age_z", "age_z2", "bmi_z", "sbp_z"],
    "dyslipidemia": ["intercept", "age_z", "age_z2", "bmi_z"],
    "diabetes": ["intercept", "age_z", "age_z2", "bmi_z", "hypertension"],
    "cancer": ["intercept", "age_z", "age_z2", "smoke"],
    "cvd": ["intercept", "age_z", "age_z2", "sbp_z", "hypertension", "diabetes", "smoke"],
    "arthritis": ["intercept", "age_z", "age_z2", "bmi_z"],
    "koa": ["intercept", "age_z", "age_z2", "bmi_z", "arthritis"],
}
