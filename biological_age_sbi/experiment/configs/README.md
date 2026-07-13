# Configs

Store small, explicit experiment configs here.

Initial expected config fields:

- age prior range
- bioindicator names
- conditional simulator parameters
- training sample counts
- BayesFlow network settings
- diagnostic settings

`bioage_diagnostics_template.json` separates the complete run metadata into:

- `variant`: short human-readable experimental label
- `simulator_config`: prior, bioindicator/process model, observation model, and feature set
- `network_config`: estimator family, architecture, and predicted quantities
- `training_config`: simulation budget, optimization settings, checkpoint, and seeds
- `evaluation_config`: held-out calibration set, quantile/interval settings, PPC budget, and output paths

Copy the template for a concrete run and replace all `null` training fields. Do
not treat the existing two-batch overfit checkpoint as a scientific baseline.

## Posterior-diagnostic development variants

The initial development comparison uses four complete configs:

1. `bioage_diag_01_simple_mendeley.json`: Mendeley-only sequential Gaussian conditionals without copula or latent factors.
2. `bioage_diag_02_advanced_mendeley.json`: Mendeley-only conditionals with copula and latent factors.
3. `bioage_diag_03_advanced_reduced_noise_mendeley.json`: variant 2 with continuous residual and latent standard deviations scaled to `0.5`.
4. `bioage_diag_04_advanced_reduced_noise_combined.json`: variant 3 refitted on the combined Mendeley+NHANES Set-E baseline.

All four use the same inference-network architecture, training budget, quantile
levels, and diagnostic budget. Variant 4 necessarily changes the feature set
from Mendeley Set D to the common Set E because KOA has no comparable NHANES
variable. Config status remains `planned` until the corresponding model has
been trained and its checkpoint and evaluation artifact exist.

Notebook `04_bioage_posterior_diagnostics.ipynb` loads these four files in
order, trains any variant whose evaluation artifact is missing, and produces
the shared comparison table and figures. Set
`BIOAGE_DIAGNOSTIC_RUN_MODE=smoke` for a small `/tmp`-only check.
