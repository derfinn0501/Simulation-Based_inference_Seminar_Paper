# Data

Use this folder for dataset-derived information that grounds the simulator.

- `raw/`: original downloaded or copied dataset extracts.
- `processed/`: cleaned summaries such as age ranges, indicator ranges, trend estimates, and covariance estimates.

Do not treat data-derived summaries as simulator truth without documenting the source and preprocessing choices.

## Current External Sources

- `raw/mendeley_data.xlsx`: Fu et al. CHARLS biological-age data from Mendeley DOI `10.17632/3rv7mf5pv9.1`.
- `raw/nhanes_2017_2020/`: selected official NHANES 2017-March 2020 Pre-Pandemic XPT files.

## Current Harmonized Outputs

- `processed/mendeley_kdm8_harmonized.csv`: Mendeley rows mapped to common internal names.
- `processed/nhanes_2017_2020_kdm8_harmonized.csv`: NHANES rows with modified KDM8 biological age.
- `processed/combined_mendeley_nhanes_kdm8.csv`: vertical concat of the two harmonized tables.
- `processed/nhanes_2017_2020_kdm8_manifest.json`: source, method, and row-count metadata.
- `processed/baseline_train_*.csv` and `processed/baseline_holdout_*.csv`: deterministic train/holdout splits for the active combined-baseline feature sets.
- `processed/empirical_bioage_model_set_e_kdm8_common_lab.json`: active lab-rich empirical simulator fitted from the combined baseline.
