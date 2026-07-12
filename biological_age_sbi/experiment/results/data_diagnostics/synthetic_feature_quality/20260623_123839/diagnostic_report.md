# Synthetic-Real Feature Quality Diagnostic

Generated: 2026-06-23T12:39:19
Output directory: `/home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/synthetic_feature_quality/20260623_123839`

## 1. Executive Summary

This diagnostic compares simulator-generated observed features against held-out Mendeley observed features for each configured feature set. It combines per-feature mismatch metrics, correlation mismatch, classifier two-sample tests, PCA overlap plots, and saved recovery metrics when available.

Recovery note:
- No saved recovery result found at /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/feature_set_batch_count_grid_copula_residual_noise_1_0_latent_noise_1_0.csv. Run notebook 02 number-of-features diagnostic first, or set up explicit retraining.

Optional diagnostic note:
- UMAP skipped: ModuleNotFoundError: No module named 'umap'

## 2. Feature-Set Comparison Table
```text
                feature_set_name  number_of_features  domain_classifier_auc  domain_classifier_accuracy  domain_classifier_balanced_accuracy
              set_a_easy_non_lab                   4               0.608743                    0.579444                             0.579444
         set_b_non_lab_diagnosis                   9               0.598967                    0.568333                             0.568333
set_c_extended_non_lab_diagnosis                  11               0.629110                    0.591111                             0.591111
              set_d_lab_enriched                  18               0.771732                    0.693333                             0.693333
```

## 3. Synthetic-Real Mismatch Summary
```text
       feature_set_name comparison_type      feature  absolute_standardized_mean_difference  wasserstein_distance  ks_statistic
     set_a_easy_non_lab     age_matched        drink                               0.059962              0.029489           NaN
     set_a_easy_non_lab     age_matched          sbp                               0.025186              3.172526      0.078309
     set_a_easy_non_lab     age_matched        smoke                               0.024805              0.012123           NaN
     set_a_easy_non_lab     age_matched          bmi                               0.013127              0.358660      0.044561
     set_a_easy_non_lab   unconditional        drink                               0.060217              0.029592           NaN
     set_a_easy_non_lab   unconditional        smoke                               0.027681              0.013509           NaN
     set_a_easy_non_lab   unconditional          sbp                               0.019160              3.172488      0.076552
     set_a_easy_non_lab   unconditional          bmi                               0.010993              0.373194      0.046960
set_b_non_lab_diagnosis     age_matched          cvd                               0.081755              0.026516           NaN
set_b_non_lab_diagnosis     age_matched        drink                               0.064032              0.031488           NaN
set_b_non_lab_diagnosis     age_matched        smoke                               0.038616              0.018893           NaN
set_b_non_lab_diagnosis     age_matched          koa                               0.036997              0.011601           NaN
set_b_non_lab_diagnosis     age_matched     diabetes                               0.035846              0.008949           NaN
set_b_non_lab_diagnosis     age_matched    arthritis                               0.025535              0.012264           NaN
set_b_non_lab_diagnosis     age_matched hypertension                               0.007452              0.003315           NaN
set_b_non_lab_diagnosis     age_matched          sbp                               0.006302              2.540084      0.056679
set_b_non_lab_diagnosis     age_matched          bmi                               0.003621              0.377906      0.053696
set_b_non_lab_diagnosis   unconditional          cvd                               0.087484              0.028305           NaN
set_b_non_lab_diagnosis   unconditional        drink                               0.063474              0.031200           NaN
set_b_non_lab_diagnosis   unconditional          koa                               0.043954              0.013831           NaN
```

## 4. Classifier Two-Sample Test Results
```text
                feature_set_name comparison_type          classifier  roc_auc  accuracy  balanced_accuracy  tn  fp  fn  tp  n_test
              set_a_easy_non_lab   unconditional logistic_regression 0.496548  0.495000           0.495000 414 486 423 477    1800
              set_a_easy_non_lab   unconditional       random_forest 0.608743  0.579444           0.579444 494 406 351 549    1800
              set_a_easy_non_lab     age_matched logistic_regression 0.513088  0.505556           0.505556 386 514 376 524    1800
              set_a_easy_non_lab     age_matched       random_forest 0.618621  0.585556           0.585556 499 401 345 555    1800
         set_b_non_lab_diagnosis   unconditional logistic_regression 0.513167  0.496111           0.496111 439 461 446 454    1800
         set_b_non_lab_diagnosis   unconditional       random_forest 0.598967  0.568333           0.568333 459 441 336 564    1800
         set_b_non_lab_diagnosis     age_matched logistic_regression 0.526883  0.523889           0.523889 409 491 366 534    1800
         set_b_non_lab_diagnosis     age_matched       random_forest 0.597825  0.568889           0.568889 478 422 354 546    1800
set_c_extended_non_lab_diagnosis   unconditional logistic_regression 0.533058  0.538889           0.538889 352 548 282 618    1800
set_c_extended_non_lab_diagnosis   unconditional       random_forest 0.629110  0.591111           0.591111 484 416 320 580    1800
set_c_extended_non_lab_diagnosis     age_matched logistic_regression 0.531488  0.527778           0.527778 399 501 349 551    1800
set_c_extended_non_lab_diagnosis     age_matched       random_forest 0.626138  0.595000           0.595000 516 384 345 555    1800
              set_d_lab_enriched   unconditional logistic_regression 0.527642  0.525000           0.525000 411 489 366 534    1800
              set_d_lab_enriched   unconditional       random_forest 0.771732  0.693333           0.693333 606 294 258 642    1800
              set_d_lab_enriched     age_matched logistic_regression 0.539978  0.541667           0.541667 451 449 376 524    1800
              set_d_lab_enriched     age_matched       random_forest 0.799511  0.718889           0.718889 629 271 235 665    1800
```

## 5. Top Features Distinguishing Synthetic From Real
```text
                feature_set_name          classifier  feature  importance
              set_a_easy_non_lab       random_forest      sbp    0.528776
              set_a_easy_non_lab       random_forest      bmi    0.443734
              set_a_easy_non_lab logistic_regression      bmi    0.051089
              set_a_easy_non_lab logistic_regression      sbp    0.048950
              set_a_easy_non_lab logistic_regression    drink    0.023627
         set_b_non_lab_diagnosis       random_forest      sbp    0.467886
         set_b_non_lab_diagnosis       random_forest      bmi    0.408085
         set_b_non_lab_diagnosis logistic_regression      cvd    0.085127
         set_b_non_lab_diagnosis logistic_regression    drink    0.083291
         set_b_non_lab_diagnosis logistic_regression      sbp    0.039944
set_c_extended_non_lab_diagnosis       random_forest      sbp    0.427438
set_c_extended_non_lab_diagnosis       random_forest      bmi    0.399778
set_c_extended_non_lab_diagnosis logistic_regression      cvd    0.068370
set_c_extended_non_lab_diagnosis logistic_regression diabetes    0.063729
set_c_extended_non_lab_diagnosis logistic_regression      koa    0.061743
              set_d_lab_enriched       random_forest    hba1c    0.201601
              set_d_lab_enriched logistic_regression      cvd    0.148650
              set_d_lab_enriched       random_forest      sbp    0.121380
              set_d_lab_enriched       random_forest      crp    0.102352
              set_d_lab_enriched       random_forest      bmi    0.096005
```

## 6. Parameter Recovery Table
No rows available.

## 7. Parameters With Good Synthetic Recovery But Poor Real Recovery
No rows available.

## 8. Parameters With Poor Synthetic And Poor Real Recovery
No rows available.

## 9. Parameters With Acceptable Real Recovery
No rows available.

## 10. Cross-Feature-Set Interpretation

Use `domain_classifier_auc_vs_feature_set.png`, `recovery_vs_feature_set.png`, and `recovery_gap_vs_feature_set.png` together. A high domain AUC with high synthetic recovery and low real recovery is evidence for simulator-real mismatch. Low synthetic and real recovery suggests weak identifiability from the selected features.

## 11. Recommended Next Actions

- Inspect feature sets with high domain classifier AUC first.
- Prioritize simulator calibration for features with large standardized mean differences and high classifier importance.
- If recovery results are unavailable, run notebook `02_feature_set_data_diagnostics.ipynb` first and rerun this notebook.
- If synthetic and real are hard to distinguish but real recovery remains poor, investigate label noise, weak feature identifiability, or model capacity rather than feature-domain mismatch.