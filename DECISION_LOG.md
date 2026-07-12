# Decision Log

## 2026-06-19 - Biological-age simulator refactor

- Use Mendeley non-lab Set B variables for the first empirical simulator: BMI, systolic blood pressure, smoking, drinking, hypertension, diabetes, CVD, arthritis, and KOA.
- Fit the empirical probabilistic model in notebook `00` and save the fitted model plus train/holdout splits under `biological_age_sbi/experiment/data/processed/`.
- Keep BayesFlow notebook `01` as the readable SBI walk-through that loads the fitted simulator, runs prior predictive checks, builds the adapter, trains point estimates, validates synthetically, and checks held-out Mendeley rows.
- Model interdependence first through sequential conditional regressions. Keep latent factors and conditional copulas as later upgrades if the simple simulator under-represents dependence.

## 2026-06-19 - Add latent risk layer to biological-age simulator

- Add a small age-dependent latent risk layer between biological age and generated indicators: metabolic risk, cardiovascular risk, joint burden, and behavior risk.
- Keep the latent variables as simulator-only nuisance variables. They are sampled and available for diagnostics, but the adapter drops them before training so the network only observes non-lab indicators.
- Treat the loadings as modeling assumptions, not directly inferred biological truth. If the layer improves synthetic recovery but worsens prior predictive realism, tune loadings or replace the residual dependence model with a conditional copula.

## 2026-06-19 - Residual-noise ablation for bioage simulator

- Keep the saved empirical Mendeley JSON as the base model.
- Apply simulator ablations in notebook `01` in memory, starting with `continuous_residual_noise_0_5`.
- The first ablation halves continuous empirical residual noise for BMI and SBP while leaving observation noise and latent-factor noise unchanged. This tests whether recovery is limited by signal-to-noise rather than network capacity.

## 2026-06-19 - Combined residual and latent-noise ablation

- Keep the smaller point-estimation training budget and MLP settings as the comparison baseline.
- Change notebook `01` to the active simulator variant `residual_noise_0_5_latent_noise_0_5`.
- This variant halves BMI/SBP residual noise and halves latent-factor stochasticity while leaving observation noise, direct age effects, latent age effects, and latent loadings unchanged.
- Keep explicit in-notebook knobs for later one-at-a-time tests of direct age effect strength, latent age effect strength, and latent loading strength.

## 2026-06-22 - Add conditional Gaussian copula to bioage simulator

- Upgrade the saved empirical biological-age model to version 2: `sequential_conditionals_with_gaussian_copula`.
- Fit a conditional Gaussian copula on PIT-transformed residuals for BMI, SBP, smoking, drinking, hypertension, diabetes, CVD, arthritis, and KOA.
- Use the copula in the simulator through correlated uniforms, preserving the fitted conditional marginals while adding residual cross-indicator dependence.
- Keep `COPULA_ENABLED` as an explicit notebook `01` switch for A/B checks against the non-copula simulator.

## 2026-06-23 - Larger point-estimation run for copula simulator

- Increase notebook `01` to a larger-capacity BayesFlow point-estimation run because `real_r` improved mainly with network size.
- Use 131,072 simulated training examples, 1,000 validation examples, 40 epochs, and an MLP with widths `(512, 512, 256)`.
- Keep the simulator variant fixed to the copula-enabled residual/latent noise ablation so the run tests data and network capacity rather than another simulator change.

## 2026-06-23 - Two-batch overfit diagnostic

- Add notebook `01` approach `overfit_two_batch_memorization_check`.
- Train on only two fixed batches, 256 simulated examples total, for 300 epochs with dropout disabled.
- Omit validation data for this diagnostic. The expected result is near-collapse of training loss; failure would indicate an issue with capacity, architecture, adapter/target scaling, or gradient updates before simulator realism is blamed.

## 2026-06-23 - Batch-count learning curve diagnostic

- Add notebook `01` approach `batch_count_learning_curve`.
- Train fresh networks on 1, 2, 8, 32, 128, and 1,024 batches to test whether increasing simulated data improves learnability and transfer.
- Save final/min training loss plus synthetic and real Pearson `r` to a CSV, and plot training loss and recovery correlation versus number of simulations.
- Also save full per-epoch training-loss histories to a long-form CSV and plot convergence curves by batch count.
- Define `synthetic_r` as correlation between simulator ground-truth biological age and predicted mean; define `real_r` as correlation between held-out Mendeley biological age and predicted mean.

## 2026-06-23 - Separate data diagnostics from network diagnostics

- Keep batch-count, epoch, overfit, and architecture diagnostics under `biological_age_sbi/experiment/results/network_diagnostics/`.
- Add `biological_age_sbi/experiment/results/data_diagnostics/` for diagnostics where the observation design changes.
- Add feature-set metadata for four Mendeley observation designs: easy non-lab, current non-lab diagnosis, extended non-lab diagnosis, and lab-enriched.
- Refactor the empirical model and simulator components to read columns, true keys, observed keys, and adapter statistics from the fitted model JSON, while preserving notebook `01` Set-B compatibility.
- Add notebook `02_feature_set_data_diagnostics.ipynb` to fit/save one empirical simulator per feature set and compare synthetic/real recovery plus q10-q90 width as a certainty proxy.

## 2026-06-23 - Number-of-features batch-count grid

- Update notebook `02` so data diagnostics train one fresh network for each feature set at 2, 8, and 32 batches.
- Save all outputs under `biological_age_sbi/experiment/results/data_diagnostics/number_of_features/`.
- For every trained network, record final/min training loss, synthetic and real Pearson `r`, MAE/RMSE, q10-q90 width, training seeds, batch count, and feature count.
- Plot loss paths by feature set and batch count, training loss versus training simulations, Pearson `r` versus training simulations, and q10-q90 width versus number of features.

## 2026-06-23 - Synthetic-real feature quality diagnostic

- Add notebook `03_feature_quality_synthetic_real_mismatch.ipynb` for simulator-real feature quality checks.
- Save timestamped outputs under `biological_age_sbi/experiment/results/data_diagnostics/synthetic_feature_quality/`.
- Compare synthetic and real observed features per feature set with per-feature mismatch metrics, correlation mismatch, classifier two-sample tests, PCA overlap plots, and saved recovery metrics when available.
- Include both unconditional and age-matched classifier tests so domain gaps are not only driven by age distribution differences.
- The first completed run showed random-forest domain AUC around 0.72-0.73 for non-lab feature sets and around 0.94 for the lab-enriched set, suggesting lab features currently expose a stronger simulator-real mismatch.

## 2026-06-23 - BMI/SBP calibration targets

- Extend notebook `03_feature_quality_synthetic_real_mismatch.ipynb` with explicit continuous calibration diagnostics for BMI and SBP.
- Split calibration into three targets: marginal distribution shape, age-conditional structure, and joint BMI-SBP dependence.
- Save `continuous_shape_calibration_metrics.csv`, `age_conditional_continuous_calibration_metrics.csv`, and `joint_dependence_calibration_metrics.csv` in each timestamped synthetic-feature-quality run.
- With residual and latent noise scales restored to 1.0, BMI/SBP standard deviations are close to real data, so the remaining mismatch is less about global under-dispersion and more about shape, age-bin structure, and BMI-SBP coupling.

## 2026-06-23 - Calibrated BMI/SBP simulator variant

- Add simulator support for empirical continuous residual bootstrap pools by biological-age bin. This replaces Gaussian residual draws when explicitly enabled by a notebook variant.
- Add an explicit `sbp_bmi` effect scale for shrinking the direct BMI coefficient in the SBP conditional model.
- Add train-only age-bin SBP mean correction support, with shrinkage. The active notebook `03` variant uses half-strength correction to avoid overcorrecting sparse older bins.
- The current diagnostic variant is `copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5`.
- Compared with the prior `copula_residual_noise_1_0_latent_noise_1_0` diagnostic, this substantially reduces BMI/SBP marginal distance and BMI-SBP slope mismatch for non-lab feature sets. Remaining issues: Set B underestimates the BMI-SBP slope, and the lab-enriched feature set still has a high domain-classifier AUC driven mostly by lab variables.

## 2026-06-23 - Activate calibrated simulator in notebooks 01 and 02

- Update notebook `01_bioage_bayesflow_step_by_step.ipynb` and notebook `02_feature_set_data_diagnostics.ipynb` to use the calibrated simulator variant `copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5`.
- Notebook `02` now saves each run into a unique timestamped subfolder under `biological_age_sbi/experiment/results/data_diagnostics/number_of_features/`.
- Completed notebook `02` run `20260623_132831_840242_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5` with 12 grid rows: four feature sets times three training sizes.
- At 4,096 training simulations, real Pearson `r` was 0.462 for Set A, 0.453 for Set B, 0.440 for Set C, and 0.551 for Set D. The lab-enriched feature set currently gives the strongest real recovery.

## 2026-07-03 - Add NHANES KDM8 concat for bioage data grounding

- Confirmed the existing Mendeley `Biological Age` source is Fu et al. 2025 / Mendeley DOI `10.17632/3rv7mf5pv9.1`, based on CHARLS and computed with KDM using TC, TG, HbA1c, BUN, creatinine, SBP, hs-CRP, and platelet count.
- Use NHANES 2017-March 2020 Pre-Pandemic public XPT files as the first external dataset because they contain the same KDM8 biomarker panel plus BMI and overlapping diagnosis/lifestyle variables.
- Compute NHANES biological age as `modified_kdm8_nhanes_2017_2020_biopro` using a Python port of the BioAge `kdm_calc` formula, fit separately by sex within complete NHANES KDM8 rows age 45+.
- Save raw NHANES files under `biological_age_sbi/experiment/data/raw/nhanes_2017_2020/` and harmonized outputs under `biological_age_sbi/experiment/data/processed/`.
- Keep `source_dataset` and `bioage_method` in the combined table because Mendeley published KDM8 and NHANES recomputed KDM8 are comparable but not identical targets.

## 2026-07-03 - Make combined KDM8 data the notebook baseline

- Set `MODEL_NAME` to `set_e_kdm8_common_lab`, using the combined Mendeley+NHANES KDM8 table as the active empirical simulator baseline.
- Add `set_f_common_non_lab_diagnosis` as a KOA-free common diagnosis feature set so comparisons can use both Mendeley and NHANES rows rather than collapsing to Mendeley-only complete cases.
- Update experiment notebooks `00`-`03` to load `combined_mendeley_nhanes_kdm8.csv`, save/read `baseline_train_*` and `baseline_holdout_*` files, and derive observed/true adapter keys from the fitted empirical model.
- Use the feature-set ladder `set_a_easy_non_lab`, `set_f_common_non_lab_diagnosis`, and `set_e_kdm8_common_lab` for data-richness diagnostics.
- Keep `source_dataset` out of the current simulator state; a later experiment can treat dataset/source as a design or nuisance variable if source-specific calibration becomes necessary.


## 2026-07-04 - Add shared-server MLflow setup for bioage work

- Use the existing guided-research MLflow server as the tracking backend for seminar bioage work, but isolate this project in a separate MLflow experiment named `seminar-bioage-sbi`.
- Treat the separate MLflow experiment page as the new project webpage; do not start a second tracking server unless the shared server becomes unavailable.
- Add a small JSON config, setup script, and reusable helper functions so code-reading, simulator diagnostics, and later BayesFlow runs can log stepwise progress, metrics, and artifacts into the same experiment.
