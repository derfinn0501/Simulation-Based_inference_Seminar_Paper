# Synthetic-Real Feature Quality Diagnostic

Generated: 2026-07-04T12:16:35
Output directory: `/home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/synthetic_feature_quality/20260704_120918`

## 1. Executive Summary

This diagnostic compares simulator-generated observed features against held-out baseline observed features for each configured feature set. It combines per-feature mismatch metrics, correlation mismatch, classifier two-sample tests, PCA overlap plots, and saved recovery metrics when available.

Optional diagnostic note:
- UMAP skipped: ModuleNotFoundError: No module named 'umap'

## 2. Feature-Set Comparison Table
```text
              feature_set_name  number_of_features  domain_classifier_auc  domain_classifier_accuracy  domain_classifier_balanced_accuracy  overall_synthetic_recovery_pearson_r  overall_real_recovery_pearson_r  overall_recovery_gap
            set_a_easy_non_lab                   4               0.576332                    0.548333                             0.548333                              0.445008                         0.406601              0.038407
set_f_common_non_lab_diagnosis                  10               0.615331                    0.584444                             0.584444                              0.501594                         0.464497              0.037098
         set_e_kdm8_common_lab                  17               0.844983                    0.762778                             0.762778                              0.586406                         0.551311              0.035094
```

## 3. Synthetic-Real Mismatch Summary
```text
     feature_set_name comparison_type      feature  absolute_standardized_mean_difference  wasserstein_distance  ks_statistic
   set_a_easy_non_lab     age_matched          bmi                               0.028083              0.228569      0.026760
   set_a_easy_non_lab     age_matched        smoke                               0.026437              0.012990           NaN
   set_a_easy_non_lab     age_matched        drink                               0.016119              0.008054           NaN
   set_a_easy_non_lab     age_matched          sbp                               0.011780              1.031811      0.030657
   set_a_easy_non_lab   unconditional        smoke                               0.025237              0.012399           NaN
   set_a_easy_non_lab   unconditional          bmi                               0.020057              0.236595      0.025810
   set_a_easy_non_lab   unconditional        drink                               0.015698              0.007844           NaN
   set_a_easy_non_lab   unconditional          sbp                               0.005119              1.090059      0.028087
set_e_kdm8_common_lab     age_matched     diabetes                               0.096137              0.029870           NaN
set_e_kdm8_common_lab     age_matched          cvd                               0.033457              0.011631           NaN
set_e_kdm8_common_lab     age_matched        drink                               0.031211              0.015596           NaN
set_e_kdm8_common_lab     age_matched    platelets                               0.030573              0.015660      0.029077
set_e_kdm8_common_lab     age_matched        hba1c                               0.028743              0.008492      0.044674
set_e_kdm8_common_lab     age_matched          crp                               0.028398              0.033111      0.030135
set_e_kdm8_common_lab     age_matched       cancer                               0.027536              0.005287           NaN
set_e_kdm8_common_lab     age_matched   creatinine                               0.026232              0.009841      0.024584
set_e_kdm8_common_lab     age_matched          bmi                               0.019255              0.226133      0.028020
set_e_kdm8_common_lab     age_matched           tc                               0.017373              0.006825      0.022733
set_e_kdm8_common_lab     age_matched hypertension                               0.014120              0.006609           NaN
set_e_kdm8_common_lab     age_matched           tg                               0.012549              0.014715      0.023526
```

## 4. Continuous Calibration Targets: Shape, Age Structure, And Joint Dependence

These tables focus on continuous features present in each feature set, including the KDM8 lab biomarkers when available.

Shape calibration summary:
```text
              feature_set_name    feature  std_ratio_sim_to_real  skew_difference  q05_difference  q50_difference  q95_difference
            set_a_easy_non_lab        bmi               0.998457        -0.439475       -0.660102        0.055939        0.228699
            set_a_easy_non_lab        sbp               1.035742        -0.155549       -3.027521        1.132502       -0.136121
set_f_common_non_lab_diagnosis        bmi               0.978318        -0.612908       -0.683347        0.016931        0.403727
set_f_common_non_lab_diagnosis        sbp               1.023838        -0.108959       -1.007456        0.656672        1.419729
         set_e_kdm8_common_lab        bmi               0.978079        -0.627229       -0.663726        0.080011        0.669596
         set_e_kdm8_common_lab        sbp               1.034384        -0.133597       -2.691989        0.905482        1.275258
         set_e_kdm8_common_lab  platelets               0.967915         0.179740       -0.026720       -0.009572       -0.012606
         set_e_kdm8_common_lab        crp               0.985648        -0.082070       -0.021835       -0.024933       -0.065212
         set_e_kdm8_common_lab      hba1c               1.029079         0.074176        0.014774       -0.002776        0.036489
         set_e_kdm8_common_lab creatinine               0.932967        -0.614879        0.014947       -0.004758       -0.022612
         set_e_kdm8_common_lab        bun               1.002488         0.016174        0.018029        0.011946        0.035985
         set_e_kdm8_common_lab         tc               0.984947        -0.065112       -0.007961        0.004267       -0.008685
         set_e_kdm8_common_lab         tg               0.994336        -0.094682       -0.006372        0.022922       -0.001506
```

Joint dependence summary:
```text
              feature_set_name  corr_bmi_sbp_sim  corr_bmi_sbp_real  corr_difference  slope_sbp_on_bmi_sim  slope_sbp_on_bmi_real  slope_difference
            set_a_easy_non_lab          0.077560           0.062465         0.015095              0.310634               0.241170          0.069464
set_f_common_non_lab_diagnosis          0.092274           0.082299         0.009975              0.372045               0.317074          0.054971
         set_e_kdm8_common_lab          0.103965           0.082299         0.021666              0.423606               0.317074          0.106532
```

## 5. Classifier Two-Sample Test Results
```text
              feature_set_name comparison_type          classifier  roc_auc  accuracy  balanced_accuracy  tn  fp  fn  tp  n_test
            set_a_easy_non_lab   unconditional logistic_regression 0.493720  0.491111           0.491111 361 539 377 523    1800
            set_a_easy_non_lab   unconditional       random_forest 0.576332  0.548333           0.548333 513 387 426 474    1800
            set_a_easy_non_lab     age_matched logistic_regression 0.489037  0.496111           0.496111 444 456 451 449    1800
            set_a_easy_non_lab     age_matched       random_forest 0.565501  0.540000           0.540000 457 443 385 515    1800
set_f_common_non_lab_diagnosis   unconditional logistic_regression 0.493433  0.504444           0.504444 310 590 302 598    1800
set_f_common_non_lab_diagnosis   unconditional       random_forest 0.615331  0.584444           0.584444 514 386 362 538    1800
set_f_common_non_lab_diagnosis     age_matched logistic_regression 0.504858  0.508889           0.508889 460 440 444 456    1800
set_f_common_non_lab_diagnosis     age_matched       random_forest 0.585374  0.552778           0.552778 501 399 406 494    1800
         set_e_kdm8_common_lab   unconditional logistic_regression 0.504751  0.501667           0.501667 437 463 434 466    1800
         set_e_kdm8_common_lab   unconditional       random_forest 0.844983  0.762778           0.762778 672 228 199 701    1800
         set_e_kdm8_common_lab     age_matched logistic_regression 0.504331  0.493333           0.493333 420 480 432 468    1800
         set_e_kdm8_common_lab     age_matched       random_forest 0.863319  0.801111           0.801111 707 193 165 735    1800
```

## 6. Top Features Distinguishing Synthetic From Real
```text
              feature_set_name          classifier    feature  importance
            set_a_easy_non_lab       random_forest        bmi    0.489471
            set_a_easy_non_lab       random_forest        sbp    0.485940
            set_a_easy_non_lab logistic_regression      smoke    0.063237
            set_a_easy_non_lab logistic_regression      drink    0.019694
            set_a_easy_non_lab       random_forest      drink    0.015532
         set_e_kdm8_common_lab       random_forest      hba1c    0.213114
         set_e_kdm8_common_lab       random_forest        bmi    0.097622
         set_e_kdm8_common_lab       random_forest        sbp    0.092550
         set_e_kdm8_common_lab       random_forest        crp    0.091405
         set_e_kdm8_common_lab       random_forest creatinine    0.089963
set_f_common_non_lab_diagnosis       random_forest        bmi    0.420990
set_f_common_non_lab_diagnosis       random_forest        sbp    0.404835
set_f_common_non_lab_diagnosis logistic_regression   diabetes    0.118928
set_f_common_non_lab_diagnosis logistic_regression        bmi    0.066143
set_f_common_non_lab_diagnosis logistic_regression        cvd    0.055532
```

## 7. Parameter Recovery Table
```text
          timestamp                       approach_name          run_timestamp                                                                                                                                                                                           run_output_dir                    feature_set                                                                                                                                                                                                                                                                                                                           feature_set_description                                          simulator_variant  model_version                                 model_family  feature_count  continuous_count  binary_count  batch_count  num_training_sims  num_validation_sims  epochs  batch_size      mlp_widths mlp_norm  mlp_dropout  mlp_residual  training_seed  validation_seed  network_seed  synthetic_eval_seed loss_key  first_train_loss  final_train_loss  min_train_loss  synthetic_r  synthetic_mae  synthetic_rmse  synthetic_mean_error  synthetic_q10_q90_coverage  synthetic_mean_q10_q90_width   real_r  real_mae  real_rmse  real_mean_error  real_q10_q90_coverage  real_mean_q10_q90_width      parameter  synthetic_r_j  real_r_j  recovery_gap_j  good_synthetic_threshold  poor_threshold  acceptable_real_threshold        recovery_category
2026-07-04T12:07:51 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5             set_a_easy_non_lab                                                                                                                                                                                                                                                                     Easy non-lab indicators: BMI, systolic blood pressure, smoking, and drinking. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula              4                 2             2            8               1024                 1000     200         128 (512, 512, 256)    layer         0.05          True           1335             1435          1535                 1635     loss          0.730861          0.533740        0.358901     0.416300       7.604467        9.439783             -0.373589                       0.748                     22.467695 0.333158  7.947916   9.943332        -0.155586               0.738613                22.714146 biological_age       0.416300  0.333158        0.083143                       0.7             0.3                        0.5    mixed_or_intermediate
2026-07-04T12:11:57 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5 set_f_common_non_lab_diagnosis                                                                        Common non-lab diagnosis-expanded set for the combined Mendeley+NHANES data: BMI, systolic blood pressure, smoking, drinking, hypertension, dyslipidemia, diabetes, cancer, CVD, and arthritis. KOA is excluded because NHANES does not provide a comparable KOA variable. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula             10                 2             8            8               1024                 1000     200         128 (512, 512, 256)    layer         0.05          True           1338             1438          1538                 1638     loss          0.872071          0.284266        0.207609     0.470882       7.831180       10.138001              0.284271                       0.582                     16.113547 0.417856  8.046113  10.258239        -0.288249               0.588823                16.234167 biological_age       0.470882  0.417856        0.053026                       0.7             0.3                        0.5    mixed_or_intermediate
2026-07-04T12:10:10 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5             set_a_easy_non_lab                                                                                                                                                                                                                                                                     Easy non-lab indicators: BMI, systolic blood pressure, smoking, and drinking. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula              4                 2             2           32               4096                 1000     120         128 (512, 512, 256)    layer         0.05          True           1336             1436          1536                 1636     loss          0.654269          0.470524        0.379828     0.445008       7.236429        8.976185              0.625576                       0.788                     22.988477 0.406601  7.668631   9.558269         0.158989               0.763158                23.144967 biological_age       0.445008  0.406601        0.038407                       0.7             0.3                        0.5    mixed_or_intermediate
2026-07-04T12:14:29 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5 set_f_common_non_lab_diagnosis                                                                        Common non-lab diagnosis-expanded set for the combined Mendeley+NHANES data: BMI, systolic blood pressure, smoking, drinking, hypertension, dyslipidemia, diabetes, cancer, CVD, and arthritis. KOA is excluded because NHANES does not provide a comparable KOA variable. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula             10                 2             8           32               4096                 1000     120         128 (512, 512, 256)    layer         0.05          True           1339             1439          1539                 1639     loss          0.569637          0.362994        0.288382     0.501594       7.486165        9.526982             -0.448640                       0.718                     20.356177 0.464497  7.541238   9.556531        -0.296381               0.715201                20.216802 biological_age       0.501594  0.464497        0.037098                       0.7             0.3                        0.5    mixed_or_intermediate
2026-07-04T12:15:47 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5          set_e_kdm8_common_lab KDM8 common lab-rich baseline for the combined Mendeley+NHANES data: BMI, systolic blood pressure, platelet count, CRP, HbA1c, creatinine, BUN, total cholesterol, triglycerides, smoking, drinking, hypertension, dyslipidemia, diabetes, cancer, CVD, and arthritis. KOA is excluded because NHANES does not provide a comparable KOA variable. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula             17                 9             8            2                256                 1000     300         128 (512, 512, 256)    layer         0.05          True           1340             1440          1540                 1640     loss          1.774644          0.038540        0.033846     0.586406       7.694236        9.751710              0.286706                       0.226                      4.984020 0.551311  7.687766   9.584564         0.372875               0.193540                 4.866655 biological_age       0.586406  0.551311        0.035094                       0.7             0.3                        0.5          acceptable_real
2026-07-04T12:11:00 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5 set_f_common_non_lab_diagnosis                                                                        Common non-lab diagnosis-expanded set for the combined Mendeley+NHANES data: BMI, systolic blood pressure, smoking, drinking, hypertension, dyslipidemia, diabetes, cancer, CVD, and arthritis. KOA is excluded because NHANES does not provide a comparable KOA variable. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula             10                 2             8            2                256                 1000     300         128 (512, 512, 256)    layer         0.05          True           1337             1437          1537                 1637     loss          1.185387          0.161763        0.118570     0.324305       8.887600       11.444047             -1.020905                       0.404                     11.653020 0.293790  9.160545  11.598384        -1.020161               0.408870                12.020143 biological_age       0.324305  0.293790        0.030515                       0.7             0.3                        0.5    mixed_or_intermediate
2026-07-04T12:06:57 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5             set_a_easy_non_lab                                                                                                                                                                                                                                                                     Easy non-lab indicators: BMI, systolic blood pressure, smoking, and drinking. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula              4                 2             2            2                256                 1000     300         128 (512, 512, 256)    layer         0.05          True           1334             1434          1534                 1634     loss          1.776829          0.368645        0.331365     0.252061       8.832610       11.059788             -1.820041                       0.626                     19.458367 0.271541  8.433597  10.752573        -1.242644               0.655111                19.692194 biological_age       0.252061  0.271541       -0.019480                       0.7             0.3                        0.5 poor_synthetic_poor_real
```

## 8. Parameters With Good Synthetic Recovery But Poor Real Recovery
No rows available.

## 9. Parameters With Poor Synthetic And Poor Real Recovery
```text
          timestamp                       approach_name          run_timestamp                                                                                                                                                                                           run_output_dir        feature_set                                                       feature_set_description                                          simulator_variant  model_version                                 model_family  feature_count  continuous_count  binary_count  batch_count  num_training_sims  num_validation_sims  epochs  batch_size      mlp_widths mlp_norm  mlp_dropout  mlp_residual  training_seed  validation_seed  network_seed  synthetic_eval_seed loss_key  first_train_loss  final_train_loss  min_train_loss  synthetic_r  synthetic_mae  synthetic_rmse  synthetic_mean_error  synthetic_q10_q90_coverage  synthetic_mean_q10_q90_width   real_r  real_mae  real_rmse  real_mean_error  real_q10_q90_coverage  real_mean_q10_q90_width      parameter  synthetic_r_j  real_r_j  recovery_gap_j  good_synthetic_threshold  poor_threshold  acceptable_real_threshold        recovery_category
2026-07-04T12:06:57 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5 set_a_easy_non_lab Easy non-lab indicators: BMI, systolic blood pressure, smoking, and drinking. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula              4                 2             2            2                256                 1000     300         128 (512, 512, 256)    layer         0.05          True           1334             1434          1534                 1634     loss          1.776829          0.368645        0.331365     0.252061        8.83261       11.059788             -1.820041                       0.626                     19.458367 0.271541  8.433597  10.752573        -1.242644               0.655111                19.692194 biological_age       0.252061  0.271541        -0.01948                       0.7             0.3                        0.5 poor_synthetic_poor_real
```

## 10. Parameters With Acceptable Real Recovery
```text
          timestamp                       approach_name          run_timestamp                                                                                                                                                                                           run_output_dir           feature_set                                                                                                                                                                                                                                                                                                                           feature_set_description                                          simulator_variant  model_version                                 model_family  feature_count  continuous_count  binary_count  batch_count  num_training_sims  num_validation_sims  epochs  batch_size      mlp_widths mlp_norm  mlp_dropout  mlp_residual  training_seed  validation_seed  network_seed  synthetic_eval_seed loss_key  first_train_loss  final_train_loss  min_train_loss  synthetic_r  synthetic_mae  synthetic_rmse  synthetic_mean_error  synthetic_q10_q90_coverage  synthetic_mean_q10_q90_width   real_r  real_mae  real_rmse  real_mean_error  real_q10_q90_coverage  real_mean_q10_q90_width      parameter  synthetic_r_j  real_r_j  recovery_gap_j  good_synthetic_threshold  poor_threshold  acceptable_real_threshold recovery_category
2026-07-04T12:15:47 number_of_features_batch_count_grid 20260704_120302_399005 /home/finn/Documents/1-projects/2026-seminar/biological_age_sbi/experiment/results/data_diagnostics/number_of_features/20260704_120302_399005_copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5 set_e_kdm8_common_lab KDM8 common lab-rich baseline for the combined Mendeley+NHANES data: BMI, systolic blood pressure, platelet count, CRP, HbA1c, creatinine, BUN, total cholesterol, triglycerides, smoking, drinking, hypertension, dyslipidemia, diabetes, cancer, CVD, and arthritis. KOA is excluded because NHANES does not provide a comparable KOA variable. copula_empirical_residuals_sbp_bmi_0_5_sbp_agebin_mean_0_5              2 sequential_conditionals_with_gaussian_copula             17                 9             8            2                256                 1000     300         128 (512, 512, 256)    layer         0.05          True           1340             1440          1540                 1640     loss          1.774644           0.03854        0.033846     0.586406       7.694236         9.75171              0.286706                       0.226                       4.98402 0.551311  7.687766   9.584564         0.372875                0.19354                 4.866655 biological_age       0.586406  0.551311        0.035094                       0.7             0.3                        0.5   acceptable_real
```

## 11. Cross-Feature-Set Interpretation

Use `domain_classifier_auc_vs_feature_set.png`, `recovery_vs_feature_set.png`, and `recovery_gap_vs_feature_set.png` together. A high domain AUC with high synthetic recovery and low real recovery is evidence for simulator-real mismatch. Low synthetic and real recovery suggests weak identifiability from the selected features.

## 12. Recommended Next Actions

- Inspect feature sets with high domain classifier AUC first.
- Prioritize simulator calibration for features with large standardized mean differences and high classifier importance.
- If recovery results are unavailable, run notebook `02_feature_set_data_diagnostics.ipynb` first and rerun this notebook.
- If synthetic and real are hard to distinguish but real recovery remains poor, investigate label noise, weak feature identifiability, or model capacity rather than feature-domain mismatch.