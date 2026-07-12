# Synthetic-Real Feature Quality Diagnostic

Generated: 2026-06-23T13:14:22
Output directory: `/home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/synthetic_feature_quality/20260623_131155`

## 1. Executive Summary

This diagnostic compares simulator-generated observed features against held-out Mendeley observed features for each configured feature set. It combines per-feature mismatch metrics, correlation mismatch, classifier two-sample tests, PCA overlap plots, and saved recovery metrics when available.

Recovery note:
- No saved recovery result found at /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/feature_set_batch_count_grid_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5.csv. Run notebook 02 number-of-features diagnostic first, or set up explicit retraining.

Optional diagnostic note:
- UMAP skipped: ModuleNotFoundError: No module named 'umap'

## 2. Feature-Set Comparison Table
```text
                feature_set_name  number_of_features  domain_classifier_auc  domain_classifier_accuracy  domain_classifier_balanced_accuracy
              set_a_easy_non_lab                   4               0.566310                    0.542778                             0.542778
         set_b_non_lab_diagnosis                   9               0.565901                    0.546111                             0.546111
set_c_extended_non_lab_diagnosis                  11               0.577852                    0.551667                             0.551667
              set_d_lab_enriched                  18               0.805479                    0.727778                             0.727778
```

## 3. Synthetic-Real Mismatch Summary
```text
       feature_set_name comparison_type      feature  absolute_standardized_mean_difference  wasserstein_distance  ks_statistic
     set_a_easy_non_lab     age_matched        drink                               0.059962              0.029489           NaN
     set_a_easy_non_lab     age_matched          bmi                               0.025519              0.169064      0.019004
     set_a_easy_non_lab     age_matched        smoke                               0.024805              0.012123           NaN
     set_a_easy_non_lab     age_matched          sbp                               0.009731              0.804749      0.031455
     set_a_easy_non_lab   unconditional        drink                               0.060217              0.029592           NaN
     set_a_easy_non_lab   unconditional        smoke                               0.027681              0.013509           NaN
     set_a_easy_non_lab   unconditional          bmi                               0.023596              0.172547      0.018012
     set_a_easy_non_lab   unconditional          sbp                               0.003570              0.823638      0.030556
set_b_non_lab_diagnosis     age_matched          cvd                               0.072940              0.023533           NaN
set_b_non_lab_diagnosis     age_matched        drink                               0.064032              0.031488           NaN
set_b_non_lab_diagnosis     age_matched        smoke                               0.038616              0.018893           NaN
set_b_non_lab_diagnosis     age_matched          koa                               0.034929              0.010938           NaN
set_b_non_lab_diagnosis     age_matched     diabetes                               0.030676              0.007623           NaN
set_b_non_lab_diagnosis     age_matched    arthritis                               0.024160              0.011601           NaN
set_b_non_lab_diagnosis     age_matched          sbp                               0.018614              0.714778      0.024528
set_b_non_lab_diagnosis     age_matched          bmi                               0.015904              0.169970      0.015247
set_b_non_lab_diagnosis     age_matched hypertension                               0.014992              0.006629           NaN
set_b_non_lab_diagnosis   unconditional          cvd                               0.078940              0.025410           NaN
set_b_non_lab_diagnosis   unconditional        drink                               0.063474              0.031200           NaN
set_b_non_lab_diagnosis   unconditional        smoke                               0.042119              0.020585           NaN
```

## 4. Continuous Calibration Targets: Shape, Age Structure, And Joint Dependence

These tables focus on BMI and SBP because they are continuous features and were strong domain-classifier signals.

Shape calibration summary:
```text
                feature_set_name feature  std_ratio_sim_to_real  skew_difference  q05_difference  q50_difference  q95_difference
              set_a_easy_non_lab     bmi               0.949631        -0.984892       -0.212071       -0.164856        0.199546
              set_a_easy_non_lab     sbp               1.018665        -0.127213       -2.689843        0.177930       -1.008601
         set_b_non_lab_diagnosis     bmi               0.955946        -0.958053       -0.164111       -0.052483        0.141161
         set_b_non_lab_diagnosis     sbp               1.000035        -0.068656       -3.520684       -0.473315       -1.253483
set_c_extended_non_lab_diagnosis     bmi               0.937606        -1.052829       -0.209342        0.150713        0.142998
set_c_extended_non_lab_diagnosis     sbp               1.038881        -0.115433       -3.747548        0.424032        1.027929
              set_d_lab_enriched     bmi               0.946683        -0.997643       -0.155326       -0.105623        0.051502
              set_d_lab_enriched     sbp               1.048598        -0.083829       -3.451072        0.142537        1.722271
```

Joint dependence summary:
```text
                feature_set_name  corr_bmi_sbp_sim  corr_bmi_sbp_real  corr_difference  slope_sbp_on_bmi_sim  slope_sbp_on_bmi_real  slope_difference
              set_a_easy_non_lab          0.141110           0.148797        -0.007687              0.804543               0.790878          0.013665
         set_b_non_lab_diagnosis          0.110654           0.148797        -0.038144              0.615266               0.790878         -0.175613
set_c_extended_non_lab_diagnosis          0.143814           0.148797        -0.004983              0.846956               0.790878          0.056078
              set_d_lab_enriched          0.163150           0.148797         0.014353              0.960519               0.790878          0.169640
```

## 5. Classifier Two-Sample Test Results
```text
                feature_set_name comparison_type          classifier  roc_auc  accuracy  balanced_accuracy  tn  fp  fn  tp  n_test
              set_a_easy_non_lab   unconditional logistic_regression 0.495041  0.502778           0.502778 435 465 430 470    1800
              set_a_easy_non_lab   unconditional       random_forest 0.566310  0.542778           0.542778 469 431 392 508    1800
              set_a_easy_non_lab     age_matched logistic_regression 0.510743  0.510000           0.510000 385 515 367 533    1800
              set_a_easy_non_lab     age_matched       random_forest 0.553744  0.532222           0.532222 477 423 419 481    1800
         set_b_non_lab_diagnosis   unconditional logistic_regression 0.515075  0.504444           0.504444 439 461 431 469    1800
         set_b_non_lab_diagnosis   unconditional       random_forest 0.565901  0.546111           0.546111 459 441 376 524    1800
         set_b_non_lab_diagnosis     age_matched logistic_regression 0.529279  0.528889           0.528889 433 467 381 519    1800
         set_b_non_lab_diagnosis     age_matched       random_forest 0.563142  0.548889           0.548889 480 420 392 508    1800
set_c_extended_non_lab_diagnosis   unconditional logistic_regression 0.530759  0.526667           0.526667 339 561 291 609    1800
set_c_extended_non_lab_diagnosis   unconditional       random_forest 0.577852  0.551667           0.551667 474 426 381 519    1800
set_c_extended_non_lab_diagnosis     age_matched logistic_regression 0.525783  0.509444           0.509444 382 518 365 535    1800
set_c_extended_non_lab_diagnosis     age_matched       random_forest 0.571356  0.540000           0.540000 474 426 402 498    1800
              set_d_lab_enriched   unconditional logistic_regression 0.524668  0.515556           0.515556 395 505 367 533    1800
              set_d_lab_enriched   unconditional       random_forest 0.805479  0.727778           0.727778 657 243 247 653    1800
              set_d_lab_enriched     age_matched logistic_regression 0.540588  0.538889           0.538889 453 447 383 517    1800
              set_d_lab_enriched     age_matched       random_forest 0.853263  0.772778           0.772778 693 207 202 698    1800
```

## 6. Top Features Distinguishing Synthetic From Real
```text
                feature_set_name          classifier  feature  importance
              set_a_easy_non_lab       random_forest      sbp    0.503569
              set_a_easy_non_lab       random_forest      bmi    0.471511
              set_a_easy_non_lab logistic_regression      bmi    0.060697
              set_a_easy_non_lab logistic_regression      sbp    0.032731
              set_a_easy_non_lab logistic_regression    drink    0.023720
         set_b_non_lab_diagnosis       random_forest      sbp    0.458942
         set_b_non_lab_diagnosis       random_forest      bmi    0.424733
         set_b_non_lab_diagnosis logistic_regression    drink    0.083478
         set_b_non_lab_diagnosis logistic_regression      cvd    0.080216
         set_b_non_lab_diagnosis logistic_regression      sbp    0.044920
set_c_extended_non_lab_diagnosis       random_forest      bmi    0.423757
set_c_extended_non_lab_diagnosis       random_forest      sbp    0.401894
set_c_extended_non_lab_diagnosis logistic_regression      cvd    0.070310
set_c_extended_non_lab_diagnosis logistic_regression      koa    0.062237
set_c_extended_non_lab_diagnosis logistic_regression diabetes    0.058187
              set_d_lab_enriched       random_forest    hba1c    0.204916
              set_d_lab_enriched logistic_regression      cvd    0.157890
              set_d_lab_enriched       random_forest      crp    0.101139
              set_d_lab_enriched       random_forest      sbp    0.097682
              set_d_lab_enriched       random_forest       tg    0.090053
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