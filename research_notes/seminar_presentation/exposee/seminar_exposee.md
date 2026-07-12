# Simulation-Based Inference as Amortized Posterior Learning: From Single Simulators to Context-Conditioned Inference Models

## Seminar Context

The seminar focuses on the theoretical foundations of amortization, meta-learning, and in-context learning. My presentation uses Simulation-Based Inference (SBI) as a concrete case study for these ideas.

The central goal is to explain SBI in a way that makes its connection to amortized learning explicit. I also use my own biological-age simulation attempt as a small exploratory example, and I end with a speculative outlook toward context-conditioned posterior models that could connect SBI more directly to meta-learning and in-context learning.

## Central Question

How should we understand Simulation-Based Inference from the perspective of amortized learning?

More specifically:

Can SBI be interpreted not only as a way to approximate likelihood-free posteriors, but also as a framework for learning reusable inference procedures?

## Core Thesis

SBI was originally motivated by the problem of doing Bayesian inference when the likelihood is unavailable or intractable, but simulation is possible. In this setting, neural networks are trained to approximate posterior distributions from simulated examples.

This creates amortization: after training, the network can reuse the information learned from many simulations to infer parameters for new observations without rerunning a full inference algorithm from scratch.

My working interpretation is that amortization emerged partly as a practical consequence of building neural posterior approximators for likelihood-free inference. However, once introduced, amortization becomes one of the central conceptual advantages of SBI. It shifts inference from solving one posterior problem at a time toward learning an inference mechanism that can be reused across observations.

The speculative extension is that future SBI systems could amortize not only across observations from one simulator, but across related simulator families. This would move SBI closer to meta-learning and in-context learning.    

## Presentation Structure

### 1. Motivation: Why Simulation-Based Inference?

Many scientific models are easy to simulate but hard to write down as explicit likelihoods. This is common in ecology, epidemiology, neuroscience, economics, agent-based modeling, and other mechanistic modeling settings.

Classical Bayesian inference requires the posterior

```text
p(θ | x) proportional to p(x | θ) p(θ)
```

but this becomes difficult when `p(x | θ)` is unavailable, expensive, or analytically intractable.

SBI addresses this problem by replacing direct likelihood evaluation with simulation. Instead of calculating the likelihood, we simulate parameter-data pairs and train a neural model to learn the inverse mapping from observed data to plausible parameters.

This gives the first bridge to amortization: the system does not only infer parameters for one dataset; it learns from many simulated datasets how inference works in this model class.

### 2. Basic Components of SBI

I introduce SBI through its generative components.

#### 2.1 Prior

The prior defines which parameter values are plausible before observing data.

In notation:

```text
θ ~ p(θ)
```

The prior is important because it defines the simulation space. If the prior is unrealistic, the network learns inference on unrealistic worlds.

#### 2.2 Process Model / Ecology Model

The process model describes how the hidden system evolves or produces its ground-truth state.

In an ecological example, this could be an equation-based model of predator-prey dynamics. In my biological-age attempt, it is a model that links latent biological age to observable bioindicators.

This component defines the scientific assumptions of the simulator.

#### 2.3 Observation Model

The observation model describes how the true system state becomes measured data.

It includes measurement noise, sparse sampling, missingness, and other observational distortions. This is crucial because real data usually do not reveal the full state of the system directly.

#### 2.4 Simulator

Together, prior, process model, and observation model form the simulator:

```text
θ ~ p(θ)
x ~ pₛᵢₘ(x | θ)
```

The simulator generates synthetic pairs `(θ, x)`. These pairs are the training data for the inference network.

#### 2.5 Inference Network

The inference network learns the inverse direction:

```text
x -> posterior over θ
```

Instead of explicitly evaluating the likelihood, the network approximates a posterior distribution or posterior summary from simulated examples.

The architecture and loss function determine what kind of posterior approximation is learned.

### 3. Important Design Choices in SBI

SBI is not one fixed algorithm. It is a family of methods that differ in several design choices.

#### 3.1 Architecture Choice

The network must be adapted to the structure of the observed data.

Examples:

- MLPs for tabular or cross-sectional data
- CNNs for image-like data
- RNNs, transformers, or time-series networks for sequential data
- permutation-invariant networks for unordered sets

The architecture should reflect the structure of the observation `x`.

#### 3.2 Posterior Approximation Choice

Different SBI methods learn different objects.

Examples:

- Neural Posterior Estimation learns `qᵩ(θ | x)`
- Neural Likelihood Estimation learns `qᵩ(x | θ)`
- Neural Ratio Estimation learns likelihood-to-evidence or likelihood-ratio objects
- Flow-based methods represent flexible posterior distributions using invertible transformations
- Flow matching and diffusion-based methods learn generative dynamics that transform simple noise into posterior samples

This part is where I introduce competing or related methods such as ABC, MCMC, and likelihood-based Bayesian inference.

The contrast is:

- MCMC usually targets one posterior for one observed dataset.
- ABC avoids the explicit likelihood but often requires repeated simulation and distance thresholds.
- SBI uses simulations to train a reusable inference network.

### 4. Running Example: Lotka-Volterra

I use the Lotka-Volterra SBI instance as example of SBI.

#### 4.1 Model Components

The hidden parameters describe ecological interaction rates, such as prey growth, predation, predator death, and predator reproduction.

The process model is the Lotka-Volterra dynamical system. It defines how prey and predator populations evolve over time.

The observation model turns the full trajectories into sparse and noisy observations.

The simulator therefore produces many possible ecological worlds: each world has its own parameters and its own observed predator-prey data.

#### 4.2 Inference Task

The inference network receives observed trajectories and tries to infer the hidden parameters.

This example is useful because the distinction between hidden parameters, latent trajectories, and observed data is very clear.

#### 4.3 Critical Evaluation

The Lotka-Volterra example is strong as a teaching example because the simulator is transparent and the parameters are interpretable.

However, it is also idealized. The model structure is known, the simulator is trusted, and the observation process is explicitly controlled. This is not always true in real scientific applications.

This motivates the question: what happens when the simulator itself is only an approximation of reality?

### 5. Own Exploratory Attempt: Biological-Age SBI

I then introduce my own exploratory project.

The goal is to simulate simple biological indicators conditional on latent biological age and to train an SBI model that recovers biological age from observable features.

The intended structure mirrors the SBI setup:

- prior over plausible biological ages
- bioindicator model linking biological age to observable features
- observation model representing noise and measurement variability
- inference network predicting biological age or its posterior from observed indicators

The main conceptual difficulty is that biological age is itself not directly observed. In the dataset, biological age is already an estimated target. Therefore, the model does not recover a true biological age ground truth. It learns to recover a proxy target constructed by a previous estimation procedure.

This is an important limitation, but also a useful teaching point: SBI depends strongly on what is treated as ground truth inside the simulator.

#### 5.1 What This Attempt Shows

The biological-age example shows that SBI is not only about choosing a neural network. The simulator is central.

If the simulator does not reproduce the real feature distribution well, then good synthetic recovery may not transfer to real data.

This motivates diagnostics such as:

- synthetic parameter recovery
- real-data recovery against proxy labels
- simulator-real feature mismatch
- posterior calibration
- uncertainty estimates

The main lesson is that posterior plots alone are not enough. One must ask whether the simulator, the target, and the observations are scientifically meaningful.

### 6. Amortization in SBI

This is the conceptual core of the presentation.

#### 6.1 Why SBI Is Amortized

In classical posterior inference, a new observed dataset usually requires a new inference run.

For example, with MCMC we define a prior and likelihood for the new observation $xₒ$, then move through parameter space to sample from the new posterior $p(θ \mid xₒ)$. If another observation arrives, this posterior-sampling procedure has to be run again.

In amortized SBI, the expensive simulation and training phase happens before the final observation is known or before many observations are evaluated.

After training, the inference network can be reused:

```text
new xₒ -> approximate posterior qᵩ(θ | xₒ)
```

What is amortized is the repeated work of constructing a posterior from scratch for each new observation. The network has already learned, from simulated pairs $(θ, x)$, how observations map to plausible parameter distributions, so inference mostly becomes reusing this learned map.

#### 6.2 Side Effect or Central Feature?

SBI can be understood as being motivated primarily by likelihood-free inference: the goal was to approximate posteriors when likelihoods are unavailable.

From that perspective, amortization may look like a practical side effect of using neural networks.

However, once the neural posterior estimator is trained across many simulations, amortization becomes a central feature. It is no longer just a technical detail. It changes the computational structure of inference.

The key shift is:

```text
from solving one posterior
to learning an inference procedure
```

#### 6.3 Limitation of Standard Amortization

Standard SBI amortizes inference within one simulator or one simulator family.

The trained network is usually specialized to:

- one prior
- one simulator
- one parameter space
- one observation structure

This means the learned inference network is reusable, but only inside a narrow modeling world.

This limitation motivates the final speculative outlook.

### 7. Speculative Outlook: Toward Context-Conditioned Posterior Foundation Models

The larger vision is to extend SBI from single-simulator amortization to task-conditioned or context-conditioned inference.

In standard SBI, we train:

```text
qᵩ(θ | x)
```

for one simulator.

In a context-conditioned version, the model would receive a small context set of simulated examples:

```text
C = {(θ₁, x₁), ..., (θₙ, xₙ)}
```

and then infer parameters for a new observation:

```text
qᵩ(θₜₑₛₜ | xₜₑₛₜ, C)
```

The context set tells the model what kind of inverse problem it is currently solving.

This creates a direct bridge to in-context learning: the model adapts its inference behavior based on examples in the context, without necessarily being retrained for every new simulator.

It also creates a bridge to meta-learning: the model is trained across many related inference tasks and learns reusable structures of inverse inference.

#### 7.1 Minimal Version

A minimal research version would not try to solve all possible scientific simulators at once.

A simpler starting point would be:

- choose a family of related simulators
- keep the parameter dimension fixed
- generate many tasks from this simulator family
- compare task-specific SBI against pooled and context-conditioned models
- evaluate posterior calibration and recovery on held-out simulator variants

#### 7.2 Larger Vision

The larger vision would be a posterior foundation model for mechanistic simulation tasks.

Such a model would not replace simulator-specific scientific modeling. Instead, it would learn reusable inverse-inference patterns across many simulator families.

Possible extensions include:

- variable parameter spaces
- richer simulator metadata
- active simulation refinement
- posterior self-diagnosis
- context-conditioned flow matching
- transfer to unseen but related simulator families

At this stage, I frame this as a speculative outlook rather than a completed contribution.

### 8. Limitations and Critical Points

The main limitation of SBI is that the quality of inference depends on the quality of the simulator.

If the simulator is misspecified, the inference network may learn simulator-specific relationships that do not transfer to reality.

A second limitation is identifiability. Some parameters may not be recoverable from the available observations, even if the simulator is correct.

A third limitation is calibration. Good point estimates do not imply calibrated posterior uncertainty.

In the biological-age example, there is an additional target problem: biological age is not directly observed, but estimated by the dataset. Therefore, recovery against this target should be interpreted as recovery of a proxy label, not necessarily recovery of biological ground truth.

The foundation-model extension is also speculative. It raises open questions about task representation, simulator diversity, parameter-space alignment, posterior calibration, and evaluation on unseen simulator families.

### 9. Preliminary Slide Logic

The exposee can be transformed into slides with the following flow:

1. Why likelihood-free inference?
2. The SBI simulator: prior, process model, observation model
3. From simulator to training data
4. Inference network and posterior approximation
5. Method variants: NPE, flows, flow matching, diffusion
6. Comparison with ABC, MCMC, and classical likelihood-based inference
7. Lotka-Volterra as clean example
8. Critical evaluation of Lotka-Volterra
9. Biological-age SBI attempt
10. Why simulator realism matters
11. What amortization means in SBI
12. Side effect or central feature?
13. Limitation: one trained network per simulator
14. Context-conditioned posterior model idea
15. Link to meta-learning and in-context learning
16. Limitations and open questions
17. Conclusion

### 10. Closing Argument

SBI is a useful lens for this seminar because it makes amortization concrete.

It starts from a classical Bayesian problem: infer hidden parameters from observed data. But because the likelihood is unavailable, it uses simulation and neural networks to learn the inverse inference problem.

This turns inference into a learned reusable mapping. That is the core amortization step.

The open question is how far this idea can be extended. Standard SBI amortizes over observations from one simulator. A more ambitious direction is to amortize across inference tasks themselves, using context-conditioned models that learn how to infer from examples.

This is where SBI begins to connect naturally to meta-learning, in-context learning, and the idea of posterior foundation models.
