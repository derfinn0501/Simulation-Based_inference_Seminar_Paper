# Possible Contribution: Active FMPE-SBI With BO-Guided Simulation Design

## Core Idea

We propose an **active flow-matching SBI framework**.

An initial set of simulator runs is used to train a conditional vector field that transports noise to samples from an approximate posterior. The current posterior estimator is then evaluated using uncertainty, relevance, and validation diagnostics. A Bayesian optimization (BO) module selects new simulator configurations or experimental designs expected to maximally improve posterior estimation. These simulations are added to the training set, and the flow-matching posterior estimator is updated iteratively.

The key separation is:

$$
\text{FMPE learns the posterior.}
$$

$$
\text{BO chooses informative simulation or experiment designs.}
$$

BO is therefore not an alternative posterior estimator. It is a design and budget-allocation layer around FMPE.

## Central Technical Problem

The central technical problem is:

> How can posterior quality be turned into a scalar reward signal for BO?

The posterior itself is not a reward. It is a distribution:

$$
q_\phi(\theta\mid x).
$$

BO needs a scalar objective:

$$
f(z)\in\mathbb{R},
$$

where $z$ is the variable controlled by BO. In this project, $z$ should usually be a design variable $\psi$, not the physical parameter $\theta$.

Therefore, the framework needs a posterior-quality functional

$$
J[q_\phi]\rightarrow \mathbb{R}.
$$

BO should then optimize expected improvement in this functional:

$$
f(\psi)
=
J[q_{\phi_{\text{new}}}]
-
J[q_{\phi_{\text{old}}}].
$$

This is the clean formulation:

$$
\text{BO selects simulations that maximize expected posterior-quality improvement.}
$$

It should not merely select simulations that match the observation pointwise.

## Main Contribution Statement

The contribution can be formulated as:

> We combine flow-matching posterior estimation with BO-guided simulation design. FMPE performs amortized posterior inference, while BO adaptively selects simulator or experimental design variables that are expected to improve posterior learning under a limited simulation budget.

This is different from using BO to find optimal physical parameters. The inferred parameters $\theta$ remain the target of posterior inference. BO acts on design variables $\psi$, such as observation schedules, sensor settings, intervention choices, simulator fidelity, noise levels, or domain-randomization settings.

## Why This Is A Useful Contribution

FMPE improves the posterior estimator by replacing discrete normalizing flows with a continuous vector field. However, FMPE still needs simulator data. If simulator calls are expensive, the question becomes:

$$
\text{Which simulations should we run?}
$$

The proposed answer is:

$$
\text{Use BO to choose the most informative simulation designs.}
$$

This gives a clean division of labor:

- FMPE handles flexible posterior learning.
- BO handles efficient simulation design.
- Diagnostics provide feedback about where the current posterior estimator is weak.

## What Is New

### 1. Active Simulation Design For FMPE

Standard FMPE trains on a fixed simulation dataset:

$$
\mathcal{D}_0
=
\{(\theta_i,x_i)\}_{i=1}^{N}.
$$

The proposed framework makes this active:

$$
\mathcal{D}_{t+1}
=
\mathcal{D}_t
\cup
\{(\theta_t,\psi_t,x_t)\}.
$$

The new part is the policy that chooses $\psi_t$ using BO.

### 2. BO Over Design Variables, Not Physical Parameters

Physical parameters should usually still be sampled from the prior:

$$
\theta_t \sim p(\theta).
$$

BO chooses design variables:

$$
\psi_t
=
\arg\max_{\psi\in\Psi} a_t(\psi).
$$

The simulator is then run as

$$
x_t
\sim
p_{\text{sim}}(x\mid \theta_t,\psi_t).
$$

This avoids changing the parameter sampling distribution away from the prior. If BO directly selected $\theta$, then the posterior estimator would be trained on an adaptive proposal rather than the prior, which can bias the posterior unless corrected with sequential SBI or importance weighting.

### 3. Diagnostics As Feedback For BO

The BO objective should be built from quantities that reflect posterior learning quality. Possible feedback signals include:

- improvement in held-out posterior log probability,
- improvement in calibration or coverage,
- posterior predictive fit,
- expected information gain,
- ensemble disagreement,
- relevance of simulations to the observed data.

Thus BO is not optimizing the posterior value itself. It is optimizing the **expected usefulness of the next design**.

## Iterative Framework

### Step 0: Initial Simulations

Sample an initial dataset using a simple, non-adaptive design:

$$
\theta_i \sim p(\theta),
\qquad
\psi_i \sim p_0(\psi),
\qquad
x_i \sim p_{\text{sim}}(x\mid \theta_i,\psi_i).
$$

This gives

$$
\mathcal{D}_0
=
\{(\theta_i,\psi_i,x_i)\}_{i=1}^{N_0}.
$$

### Step 1: Train FMPE

Train a conditional vector field

$$
v_{\phi,t,x,\psi}(\theta)
$$

that transports base noise to posterior samples. The posterior estimator is

$$
q_\phi(\theta\mid x,\psi)
\approx
p(\theta\mid x,\psi).
$$

If $\psi$ is fixed or only used during training, the estimator can be simplified to

$$
q_\phi(\theta\mid x).
$$

### Step 2: Evaluate The Current Estimator

Evaluate the current FMPE model using diagnostics such as:

$$
H(q_\phi(\theta\mid x_o,\psi)),
$$

posterior predictive mismatch,

$$
d(S(\tilde{x}),S(x_o)),
$$

or calibration error,

$$
\operatorname{ECE}(q_\phi).
$$

The goal is to identify where the current posterior estimator is uncertain, poorly calibrated, or insufficiently relevant to the observation.

### Step 3: BO Selects A New Design

Fit a BO surrogate to a scalar utility:

$$
g_t(\psi)
\approx
u_t(\psi).
$$

Then select the next design:

$$
\psi_{t+1}
=
\arg\max_{\psi\in\Psi} a_t(\psi;g_t).
$$

### Step 4: Run New Simulations

Sample parameters from the prior and simulate under the BO-selected design:

$$
\theta_{t+1} \sim p(\theta),
\qquad
x_{t+1}
\sim
p_{\text{sim}}(x\mid \theta_{t+1},\psi_{t+1}).
$$

Update the dataset:

$$
\mathcal{D}_{t+1}
=
\mathcal{D}_t
\cup
\{(\theta_{t+1},\psi_{t+1},x_{t+1})\}.
$$

### Step 5: Update FMPE

Retrain or fine-tune the FMPE model on the updated dataset:

$$
q_{\phi_{t+1}}(\theta\mid x,\psi).
$$

Repeat until the simulation budget is exhausted or posterior diagnostics stop improving.

## Possible BO Feedback Functions

### Ideal But Inaccessible Reward

Ideally, a new design would be scored by how much it improves the approximation to the true posterior:

$$
u_t(\psi)
=
D_{\text{KL}}
\left(
p(\theta\mid x_o)
\|q_{\phi_t}(\theta\mid x_o)
\right)
-
D_{\text{KL}}
\left(
p(\theta\mid x_o)
\|q_{\phi_{t+1}}(\theta\mid x_o)
\right).
$$

This is conceptually correct but unavailable in real SBI because the true posterior $p(\theta\mid x_o)$ is unknown. The practical contribution is therefore to define useful proxy rewards.

### 1. Validation Log-Posterior Improvement

For simulated validation pairs, the true generating parameter is known:

$$
(\theta_i,x_i)\in\mathcal{V}.
$$

The current posterior estimator can be evaluated by the log probability it assigns to the true parameter:

$$
J_{\text{LP}}[q_\phi]
=
\frac{1}{|\mathcal{V}|}
\sum_{(\theta_i,x_i)\in\mathcal{V}}
\log q_\phi(\theta_i\mid x_i).
$$

The BO reward is the improvement:

$$
u_t(\psi)
=
J_{\text{LP}}[q_{\phi_{t+1}}]
-
J_{\text{LP}}[q_{\phi_t}].
$$

This is one of the cleanest proxy rewards because it directly asks whether the updated posterior estimator assigns higher probability to known ground-truth parameters on held-out simulations.

### 2. Observation-Relevance Utility

Choose designs whose prior-predictive simulations are close to the observed data:

$$
u_t(\psi)
=
-
\mathbb{E}_{\theta\sim p(\theta),\,x\sim p_{\text{sim}}(\cdot\mid\theta,\psi)}
\left[
d(S(x),S(x_o))
\right].
$$

This is useful when only some simulator designs generate data in the region relevant to $x_o$.

However, this should not be the central reward by itself. It rewards pointwise similarity to $x_o$, not posterior correctness. If many different parameters can produce similar observations, this objective may collapse attention onto one good-fitting region and miss a broad or multimodal posterior.

### 3. Posterior-Predictive Utility

Choose designs whose posterior predictive simulations match the observation:

$$
\theta^{(m)}
\sim
q_{\phi_t}(\theta\mid x_o,\psi),
$$

$$
\tilde{x}^{(m)}
\sim
p_{\text{sim}}(x\mid\theta^{(m)},\psi),
$$

$$
u_t(\psi)
=
-
d(S(\tilde{x}),S(x_o)).
$$

This directly connects BO to the validation logic of SBI.

Posterior predictive fit is useful but insufficient alone: a wrong posterior can sometimes generate plausible data if the inverse problem is non-identifiable.

### 4. Calibration-Improvement Utility

Choose designs that are expected to improve calibration:

$$
u_t(\psi)
=
\operatorname{ECE}(q_{\phi_t})
-
\operatorname{ECE}(q_{\phi_{t+1}}).
$$

In practice, this may require approximating the post-update effect with a surrogate, a small validation set, or periodic retraining rounds.

More explicitly, if $C_\alpha(x)$ is an $\alpha$-credible region, the coverage error can be written as

$$
E_{\text{cov}}
=
\sum_{\alpha\in A}
\left|
\widehat{\operatorname{coverage}}(\alpha)
-
\alpha
\right|.
$$

Then the reward can be

$$
u_t(\psi)
=
E_{\text{cov},t}
-
E_{\text{cov},t+1}.
$$

This asks whether posterior uncertainty is statistically meaningful.

### 5. Ensemble-Disagreement Utility

Train an ensemble of FMPE models and choose designs where their vector fields or posterior densities disagree:

$$
u_t(\psi)
=
\mathbb{E}
\left[
\operatorname{Var}_m
\left(
\log q_{\phi_t^{(m)}}(\theta\mid x,\psi)
\right)
\right].
$$

Alternatively, use vector-field disagreement:

$$
u_t(\psi)
=
\mathbb{E}
\left[
\operatorname{Var}_m
\left(
v_{\phi_t^{(m)},t,x,\psi}(\theta)
\right)
\right].
$$

This is especially natural for FMPE because the learned object is a conditional vector field.

Disagreement alone can be dangerous because the model may be uncertain in irrelevant regions. A stronger utility combines epistemic uncertainty with relevance:

$$
u_t(\psi)
=
u_{\text{disagreement}}(\psi)
+
\lambda u_{\text{relevance}}(\psi).
$$

### 6. Information-Gain Utility

Choose designs expected to reduce posterior uncertainty:

$$
u_t(\psi)
=
I(\theta;x\mid\psi)
=
H(\theta)
-
\mathbb{E}_{x\sim p(x\mid\psi)}
\left[
H(\theta\mid x,\psi)
\right].
$$

This is conceptually clean, but may be harder to estimate robustly.

## Recommended Reward Formulation

A useful general form is:

$$
u_t(\psi)
=
\Delta J_{\text{LP}}
+
\lambda_1 \Delta J_{\text{cal}}
+
\lambda_2 \Delta J_{\text{PPC}}
+
\lambda_3 U_{\text{EIG}}.
$$

where:

- $\Delta J_{\text{LP}}$ is improvement in validation log posterior,
- $\Delta J_{\text{cal}}$ is improvement in calibration or coverage,
- $\Delta J_{\text{PPC}}$ is improvement in posterior predictive fit,
- $U_{\text{EIG}}$ is expected information gain.

For a small seminar project, the simplest defensible version is probably:

$$
u_t(\psi)
=
\Delta J_{\text{LP}}
-
\lambda E_{\text{cov}}.
$$

This keeps the reward focused on posterior correctness rather than only observation matching.

## Reward Evaluation Cost

The direct version is expensive:

$$
\psi
\rightarrow
\text{simulate}
\rightarrow
\text{update FMPE}
\rightarrow
\text{evaluate posterior quality}
\rightarrow
u_t(\psi).
$$

This is conceptually clean but computationally heavy.

A cheaper practical version is to use acquisition proxies from the current FMPE model:

$$
\psi
\rightarrow
\text{estimate informativeness}
\rightarrow
\text{BO acquisition}
\rightarrow
\text{simulate only selected designs}.
$$

For the first implementation, the project can compare both levels:

- **Direct reward:** update the model periodically and evaluate posterior-quality improvement.
- **Proxy reward:** use current posterior uncertainty, relevance, or ensemble disagreement before retraining.

## What BO Should Not Do

BO should not directly optimize the inferred physical parameters:

$$
\theta_{t+1}
=
\arg\max_\theta a_t(\theta).
$$

That changes the training distribution from

$$
\theta\sim p(\theta)
$$

to an adaptive proposal

$$
\theta\sim r_t(\theta).
$$

Then FMPE no longer learns the prior-based posterior unless the training objective is corrected, for example with importance weights

$$
w_t(\theta)
=
\frac{p(\theta)}{r_t(\theta)}.
$$

This is a valid but different project: sequential or proposal-based SBI. For the current contribution, the cleaner route is BO over design variables $\psi$.

## Evaluation Plan

Compare four systems under equal simulation budgets:

1. baseline SBI without BO,
2. baseline SBI with BO-guided design,
3. FMPE without BO,
4. FMPE with BO-guided design.

Useful metrics:

- posterior predictive fit,
- simulation budget needed to reach a target error,
- expected coverage or SBC,
- C2ST against reference posterior if available,
- validation loss,
- posterior uncertainty reduction,
- wall-clock simulation cost.

The strongest result would be:

$$
\text{FMPE + BO}
$$

achieves comparable or better posterior quality than

$$
\text{FMPE without BO}
$$

using fewer simulator calls.

## Final Position

The refined contribution is:

> Active FMPE-SBI: a framework where FMPE learns the amortized posterior through a conditional vector field, while BO adaptively selects simulator configurations or experimental designs that make each simulator call maximally informative.

This is modest, clear, and defensible: BO improves simulation efficiency; FMPE remains the posterior estimator.
