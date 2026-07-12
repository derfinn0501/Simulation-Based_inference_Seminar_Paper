# Synthetic-Real Feature Quality Diagnostic

Generated: 2026-06-23T13:06:02
Output directory: `/home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/synthetic_feature_quality/20260623_130459`

## 1. Executive Summary

This diagnostic compares simulator-generated observed features against held-out Mendeley observed features for each configured feature set. It combines per-feature mismatch metrics, correlation mismatch, classifier two-sample tests, PCA overlap plots, and saved recovery metrics when available.

Recovery note:
- No saved recovery result found at /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/feature_set_batch_count_grid_copula_empirical_residuals_sbp_bmi_0_5.csv. Run notebook 02 number-of-features diagnostic first, or set up explicit retraining.

Optional diagnostic note:
- UMAP skipped: ModuleNotFoundError: No module named 'umap'

## 2. Feature-Set Comparison Table
```text
                feature_set_name  number_of_features  domain_classifier_auc  domain_classifier_accuracy  domain_classifier_balanced_accuracy
              set_a_easy_non_lab                   4               0.567065                    0.549444                             0.549444
         set_b_non_lab_diagnosis                   9               0.561495                    0.550000                             0.550000
set_c_extended_non_lab_diagnosis                  11               0.589020                    0.569444                             0.569444
              set_d_lab_enriched                  18               0.797191                    0.716111                             0.716111
```

## 3. Synthetic-Real Mismatch Summary
```text
       feature_set_name comparison_type      feature  absolute_standardized_mean_difference  wasserstein_distance  ks_statistic
     set_a_easy_non_lab     age_matched        drink                               0.059962              0.029489           NaN
     set_a_easy_non_lab     age_matched          bmi                               0.025519              0.169064      0.019004
     set_a_easy_non_lab     age_matched        smoke                               0.024805              0.012123           NaN
     set_a_easy_non_lab     age_matched          sbp                               0.010927              1.042732      0.034731
     set_a_easy_non_lab   unconditional        drink                               0.060217              0.029592           NaN
     set_a_easy_non_lab   unconditional        smoke                               0.027681              0.013509           NaN
     set_a_easy_non_lab   unconditional          bmi                               0.023596              0.172547      0.018012
     set_a_easy_non_lab   unconditional          sbp                               0.004361              1.143416      0.033773
set_b_non_lab_diagnosis     age_matched          cvd                               0.071955              0.023202           NaN
set_b_non_lab_diagnosis     age_matched        drink                               0.064032              0.031488           NaN
set_b_non_lab_diagnosis     age_matched        smoke                               0.038616              0.018893           NaN
set_b_non_lab_diagnosis     age_matched          koa                               0.034929              0.010938           NaN
set_b_non_lab_diagnosis     age_matched     diabetes                               0.033267              0.008286           NaN
set_b_non_lab_diagnosis     age_matched    arthritis                               0.024160              0.011601           NaN
set_b_non_lab_diagnosis     age_matched          sbp                               0.022286              0.824714      0.031157
set_b_non_lab_diagnosis     age_matched          bmi                               0.015904              0.169970      0.015247
set_b_non_lab_diagnosis     age_matched hypertension                               0.009731              0.004309           NaN
set_b_non_lab_diagnosis   unconditional          cvd                               0.077986              0.025088           NaN
set_b_non_lab_diagnosis   unconditional        drink                               0.063474              0.031200           NaN
set_b_non_lab_diagnosis   unconditional        smoke                               0.042119              0.020585           NaN
```

## 4. Continuous Calibration Targets: Shape, Age Structure, And Joint Dependence

These tables focus on BMI and SBP because they are continuous features and were strong domain-classifier signals.

Shape calibration summary:
```text
                feature_set_name feature  std_ratio_sim_to_real  skew_difference  q05_difference  q50_difference  q95_difference
              set_a_easy_non_lab     bmi               0.949631        -0.984892       -0.212071       -0.164856        0.199546
              set_a_easy_non_lab     sbp               1.043780        -0.138824       -3.442403        0.254667       -0.189994
         set_b_non_lab_diagnosis     bmi               0.955946        -0.958053       -0.164111       -0.052483        0.141161
         set_b_non_lab_diagnosis     sbp               1.028465        -0.073557       -4.102926       -0.736087        0.124776
set_c_extended_non_lab_diagnosis     bmi               0.937606        -1.052829       -0.209342        0.150713        0.142998
set_c_extended_non_lab_diagnosis     sbp               1.063816        -0.118343       -4.646439        0.371480        2.236290
              set_d_lab_enriched     bmi               0.946683        -0.997643       -0.155326       -0.105623        0.051502
              set_d_lab_enriched     sbp               1.073417        -0.095716       -4.181435       -0.043223        3.265319
```

Joint dependence summary:
```text
                feature_set_name  corr_bmi_sbp_sim  corr_bmi_sbp_real  corr_difference  slope_sbp_on_bmi_sim  slope_sbp_on_bmi_real  slope_difference
              set_a_easy_non_lab          0.139507           0.148797        -0.009290              0.815013               0.790878          0.024135
         set_b_non_lab_diagnosis          0.106314           0.148797        -0.042484              0.607938               0.790878         -0.182941
set_c_extended_non_lab_diagnosis          0.140810           0.148797        -0.007987              0.849171               0.790878          0.058293
              set_d_lab_enriched          0.161328           0.148797         0.012531              0.972272               0.790878          0.181394
```

## 5. Classifier Two-Sample Test Results
```text
                feature_set_name comparison_type          classifier  roc_auc  accuracy  balanced_accuracy  tn  fp  fn  tp  n_test
              set_a_easy_non_lab   unconditional logistic_regression 0.493627  0.498889           0.498889 428 472 430 470    1800
              set_a_easy_non_lab   unconditional       random_forest 0.567065  0.549444           0.549444 488 412 399 501    1800
              set_a_easy_non_lab     age_matched logistic_regression 0.510862  0.510000           0.510000 385 515 367 533    1800
              set_a_easy_non_lab     age_matched       random_forest 0.562131  0.546111           0.546111 490 410 407 493    1800
         set_b_non_lab_diagnosis   unconditional logistic_regression 0.515528  0.502778           0.502778 439 461 434 466    1800
         set_b_non_lab_diagnosis   unconditional       random_forest 0.561495  0.550000           0.550000 449 451 359 541    1800
         set_b_non_lab_diagnosis     age_matched logistic_regression 0.529606  0.526111           0.526111 425 475 378 522    1800
         set_b_non_lab_diagnosis     age_matched       random_forest 0.564844  0.540556           0.540556 473 427 400 500    1800
set_c_extended_non_lab_diagnosis   unconditional logistic_regression 0.531825  0.525556           0.525556 343 557 297 603    1800
set_c_extended_non_lab_diagnosis   unconditional       random_forest 0.589020  0.569444           0.569444 490 410 365 535    1800
set_c_extended_non_lab_diagnosis     age_matched logistic_regression 0.527705  0.513333           0.513333 385 515 361 539    1800
set_c_extended_non_lab_diagnosis     age_matched       random_forest 0.582021  0.551111           0.551111 487 413 395 505    1800
              set_d_lab_enriched   unconditional logistic_regression 0.526393  0.518333           0.518333 398 502 365 535    1800
              set_d_lab_enriched   unconditional       random_forest 0.797191  0.716111           0.716111 647 253 258 642    1800
              set_d_lab_enriched     age_matched logistic_regression 0.540978  0.538889           0.538889 450 450 380 520    1800
              set_d_lab_enriched     age_matched       random_forest 0.846647  0.761667           0.761667 682 218 211 689    1800
```

## 6. Top Features Distinguishing Synthetic From Real
```text
                feature_set_name          classifier    feature  importance
              set_a_easy_non_lab       random_forest        sbp    0.508182
              set_a_easy_non_lab       random_forest        bmi    0.466698
              set_a_easy_non_lab logistic_regression        bmi    0.061184
              set_a_easy_non_lab logistic_regression        sbp    0.036090
              set_a_easy_non_lab logistic_regression      drink    0.023763
         set_b_non_lab_diagnosis       random_forest        sbp    0.453179
         set_b_non_lab_diagnosis       random_forest        bmi    0.426901
         set_b_non_lab_diagnosis logistic_regression      drink    0.083482
         set_b_non_lab_diagnosis logistic_regression        cvd    0.076539
         set_b_non_lab_diagnosis logistic_regression        sbp    0.049588
set_c_extended_non_lab_diagnosis       random_forest        bmi    0.420006
set_c_extended_non_lab_diagnosis       random_forest        sbp    0.404125
set_c_extended_non_lab_diagnosis logistic_regression        cvd    0.068296
set_c_extended_non_lab_diagnosis logistic_regression   diabetes    0.064290
set_c_extended_non_lab_diagnosis logistic_regression        koa    0.062624
              set_d_lab_enriched       random_forest      hba1c    0.198029
              set_d_lab_enriched logistic_regression        cvd    0.158808
              set_d_lab_enriched       random_forest        sbp    0.102607
              set_d_lab_enriched       random_forest        crp    0.097980
              set_d_lab_enriched       random_forest creatinine    0.091989
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