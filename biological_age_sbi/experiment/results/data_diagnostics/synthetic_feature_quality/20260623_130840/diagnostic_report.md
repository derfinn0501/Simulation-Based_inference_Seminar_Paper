# Synthetic-Real Feature Quality Diagnostic

Generated: 2026-06-23T13:11:07
Output directory: `/home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/synthetic_feature_quality/20260623_130840`

## 1. Executive Summary

This diagnostic compares simulator-generated observed features against held-out Mendeley observed features for each configured feature set. It combines per-feature mismatch metrics, correlation mismatch, classifier two-sample tests, PCA overlap plots, and saved recovery metrics when available.

Recovery note:
- No saved recovery result found at /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/feature_set_batch_count_grid_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean.csv. Run notebook 02 number-of-features diagnostic first, or set up explicit retraining.

Optional diagnostic note:
- UMAP skipped: ModuleNotFoundError: No module named 'umap'

## 2. Feature-Set Comparison Table
```text
                feature_set_name  number_of_features  domain_classifier_auc  domain_classifier_accuracy  domain_classifier_balanced_accuracy
              set_a_easy_non_lab                   4               0.571391                    0.547222                             0.547222
         set_b_non_lab_diagnosis                   9               0.567006                    0.550000                             0.550000
set_c_extended_non_lab_diagnosis                  11               0.570091                    0.545000                             0.545000
              set_d_lab_enriched                  18               0.806004                    0.731667                             0.731667
```

## 3. Synthetic-Real Mismatch Summary
```text
       feature_set_name comparison_type      feature  absolute_standardized_mean_difference  wasserstein_distance  ks_statistic
     set_a_easy_non_lab     age_matched        drink                               0.059962              0.029489           NaN
     set_a_easy_non_lab     age_matched          bmi                               0.025519              0.169064      0.019004
     set_a_easy_non_lab     age_matched        smoke                               0.024805              0.012123           NaN
     set_a_easy_non_lab     age_matched          sbp                               0.008356              0.805228      0.028178
     set_a_easy_non_lab   unconditional        drink                               0.060217              0.029592           NaN
     set_a_easy_non_lab   unconditional        smoke                               0.027681              0.013509           NaN
     set_a_easy_non_lab   unconditional          bmi                               0.023596              0.172547      0.018012
     set_a_easy_non_lab   unconditional          sbp                               0.002620              0.783150      0.027340
set_b_non_lab_diagnosis     age_matched          cvd                               0.073923              0.023865           NaN
set_b_non_lab_diagnosis     age_matched        drink                               0.064032              0.031488           NaN
set_b_non_lab_diagnosis     age_matched        smoke                               0.038616              0.018893           NaN
set_b_non_lab_diagnosis     age_matched          koa                               0.034929              0.010938           NaN
set_b_non_lab_diagnosis     age_matched     diabetes                               0.028074              0.006961           NaN
set_b_non_lab_diagnosis     age_matched    arthritis                               0.024160              0.011601           NaN
set_b_non_lab_diagnosis     age_matched          bmi                               0.015904              0.169970      0.015247
set_b_non_lab_diagnosis     age_matched hypertension                               0.015744              0.006961           NaN
set_b_non_lab_diagnosis     age_matched          sbp                               0.014701              0.864741      0.033146
set_b_non_lab_diagnosis   unconditional          cvd                               0.079894              0.025732           NaN
set_b_non_lab_diagnosis   unconditional        drink                               0.063474              0.031200           NaN
set_b_non_lab_diagnosis   unconditional        smoke                               0.042119              0.020585           NaN
```

## 4. Continuous Calibration Targets: Shape, Age Structure, And Joint Dependence

These tables focus on BMI and SBP because they are continuous features and were strong domain-classifier signals.

Shape calibration summary:
```text
                feature_set_name feature  std_ratio_sim_to_real  skew_difference  q05_difference  q50_difference  q95_difference
              set_a_easy_non_lab     bmi               0.949631        -0.984892       -0.212071       -0.164856        0.199546
              set_a_easy_non_lab     sbp               0.995822        -0.121424       -2.150577        0.429128       -1.872923
         set_b_non_lab_diagnosis     bmi               0.955946        -0.958053       -0.164111       -0.052483        0.141161
         set_b_non_lab_diagnosis     sbp               0.974592        -0.068878       -2.322072       -0.186053       -2.127754
set_c_extended_non_lab_diagnosis     bmi               0.937606        -1.052829       -0.209342        0.150713        0.142998
set_c_extended_non_lab_diagnosis     sbp               1.015965        -0.117490       -2.890743        0.467367        0.554003
              set_d_lab_enriched     bmi               0.946683        -0.997643       -0.155326       -0.105623        0.051502
              set_d_lab_enriched     sbp               1.026183        -0.077123       -2.641210        0.248418        0.629122
```

Joint dependence summary:
```text
                feature_set_name  corr_bmi_sbp_sim  corr_bmi_sbp_real  corr_difference  slope_sbp_on_bmi_sim  slope_sbp_on_bmi_real  slope_difference
              set_a_easy_non_lab          0.142419           0.148797        -0.006379              0.793793               0.790878          0.002915
         set_b_non_lab_diagnosis          0.114844           0.148797        -0.033953              0.622319               0.790878         -0.168559
set_c_extended_non_lab_diagnosis          0.146608           0.148797        -0.002190              0.844363               0.790878          0.053484
              set_d_lab_enriched          0.164584           0.148797         0.015787              0.948248               0.790878          0.157370
```

## 5. Classifier Two-Sample Test Results
```text
                feature_set_name comparison_type          classifier  roc_auc  accuracy  balanced_accuracy  tn  fp  fn  tp  n_test
              set_a_easy_non_lab   unconditional logistic_regression 0.496364  0.497778           0.497778 429 471 433 467    1800
              set_a_easy_non_lab   unconditional       random_forest 0.571391  0.547222           0.547222 472 428 387 513    1800
              set_a_easy_non_lab     age_matched logistic_regression 0.510526  0.510000           0.510000 385 515 367 533    1800
              set_a_easy_non_lab     age_matched       random_forest 0.558747  0.538889           0.538889 475 425 405 495    1800
         set_b_non_lab_diagnosis   unconditional logistic_regression 0.515740  0.507222           0.507222 438 462 425 475    1800
         set_b_non_lab_diagnosis   unconditional       random_forest 0.567006  0.550000           0.550000 469 431 379 521    1800
         set_b_non_lab_diagnosis     age_matched logistic_regression 0.528225  0.522778           0.522778 432 468 391 509    1800
         set_b_non_lab_diagnosis     age_matched       random_forest 0.563944  0.536111           0.536111 476 424 411 489    1800
set_c_extended_non_lab_diagnosis   unconditional logistic_regression 0.531137  0.528889           0.528889 332 568 280 620    1800
set_c_extended_non_lab_diagnosis   unconditional       random_forest 0.570091  0.545000           0.545000 482 418 401 499    1800
set_c_extended_non_lab_diagnosis     age_matched logistic_regression 0.525667  0.510000           0.510000 383 517 365 535    1800
set_c_extended_non_lab_diagnosis     age_matched       random_forest 0.569853  0.536667           0.536667 475 425 409 491    1800
              set_d_lab_enriched   unconditional logistic_regression 0.523814  0.512222           0.512222 388 512 366 534    1800
              set_d_lab_enriched   unconditional       random_forest 0.806004  0.731667           0.731667 668 232 251 649    1800
              set_d_lab_enriched     age_matched logistic_regression 0.539662  0.538889           0.538889 454 446 384 516    1800
              set_d_lab_enriched     age_matched       random_forest 0.849737  0.762222           0.762222 677 223 205 695    1800
```

## 6. Top Features Distinguishing Synthetic From Real
```text
                feature_set_name          classifier  feature  importance
              set_a_easy_non_lab       random_forest      sbp    0.505822
              set_a_easy_non_lab       random_forest      bmi    0.469227
              set_a_easy_non_lab logistic_regression      bmi    0.060163
              set_a_easy_non_lab logistic_regression      sbp    0.029082
              set_a_easy_non_lab logistic_regression    drink    0.023679
         set_b_non_lab_diagnosis       random_forest      sbp    0.449848
         set_b_non_lab_diagnosis       random_forest      bmi    0.427281
         set_b_non_lab_diagnosis logistic_regression    drink    0.083528
         set_b_non_lab_diagnosis logistic_regression      cvd    0.081154
         set_b_non_lab_diagnosis logistic_regression      koa    0.039673
set_c_extended_non_lab_diagnosis       random_forest      bmi    0.428121
set_c_extended_non_lab_diagnosis       random_forest      sbp    0.401834
set_c_extended_non_lab_diagnosis logistic_regression      cvd    0.071042
set_c_extended_non_lab_diagnosis logistic_regression      koa    0.062446
set_c_extended_non_lab_diagnosis logistic_regression diabetes    0.060193
              set_d_lab_enriched       random_forest    hba1c    0.202550
              set_d_lab_enriched logistic_regression      cvd    0.161144
              set_d_lab_enriched       random_forest      crp    0.101544
              set_d_lab_enriched       random_forest      sbp    0.095182
              set_d_lab_enriched       random_forest       tg    0.093068
```

## 7. Parameter Recovery Table
No rows available.

## 8. Parameters With Good Synthetic Recovery But Poor Real Recovery
No rows available.

## 9. Parameters With Poor Synthetic And Poor Real Recovery
No rows available.

## 10. Parameters With Acceptable Real Recovery
No rows available.

## 11. Cross-Feature-Set Interpretation

Use `domain_classifier_auc_vs_feature_set.png`, `recovery_vs_feature_set.png`, and `recovery_gap_vs_feature_set.png` together. A high domain AUC with high synthetic recovery and low real recovery is evidence for simulator-real mismatch. Low synthetic and real recovery suggests weak identifiability from the selected features.

## 12. Recommended Next Actions

- Inspect feature sets with high domain classifier AUC first.
- Prioritize simulator calibration for features with large standardized mean differences and high classifier importance.
- If recovery results are unavailable, run notebook `02_feature_set_data_diagnostics.ipynb` first and rerun this notebook.
- If synthetic and real are hard to distinguish but real recovery remains poor, investigate label noise, weak feature identifiability, or model capacity rather than feature-domain mismatch.