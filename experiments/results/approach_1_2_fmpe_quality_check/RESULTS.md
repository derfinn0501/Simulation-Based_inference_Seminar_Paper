# Approach 1.2 FMPE Quality Check Results

This diagnostic evaluates posterior estimators under random simulation design only.
It does not use BO, so it tests whether FMPE is good enough before active design is judged.

Final evaluated budget: `140` simulator calls.

## Final-Budget Summary

| Method | Range-norm RMSE | Prior-std RMSE | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: |
| prior_mean | 0.2898 | 1.0039 | 0.0139 | 1.0358 |
| gaussian_npe | 0.2309 | 0.7997 | 0.1426 | 0.8960 |
| rectified_fmpe | 0.2246 | 0.7782 | 0.2787 | 0.8065 |

## Current Verdict

At the final budget, the lightweight FMPE model is learning real signal:

- Gaussian NPE improves range-normalized RMSE over the prior mean by `20.3%`.
- Rectified FMPE improves range-normalized RMSE over the prior mean by `22.5%`.
- Rectified FMPE improves range-normalized RMSE over Gaussian NPE by `2.7%`.
- Rectified FMPE predictive RMSE is `0.8065` versus `0.8960` for Gaussian NPE.
- Rectified FMPE coverage error is `0.2787` versus `0.1426` for Gaussian NPE.

So the current wording should be:

> FMPE is better than trivial and competitive on point-estimate quality, but not yet well calibrated.

The prior-mean coverage number is low because it uses broad prior credible intervals; it should not be interpreted as useful posterior inference.


## Interpretation Rule

FMPE should only be called good if it clearly improves over the prior mean,
matches or beats the Gaussian baseline, improves with more simulations, and has usable calibration.
