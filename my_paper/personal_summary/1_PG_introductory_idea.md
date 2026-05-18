# Introductory Idea

## Intuition

Simulation-based inference (SBI) solves inverse problems for stochastic simulators: given observed data $x_o$, infer parameter values $\theta$ that could have generated it and are consistent with prior knowledge (Guide, p1:L45-L48; p2:L1-L7).

The practical motivation is that many simulators can generate samples but do not provide tractable likelihoods or gradients, which makes classical MCMC or VI difficult or impossible to apply directly (Guide, p2:L12-L17).

Modern neural SBI therefore runs the simulator many times, learns the relation between simulated data $x$ and parameters $\theta$, and then evaluates the trained inference network at $x_o$ to obtain an approximate posterior (Guide, p2:L21-L29).

The workflow is: define simulator and prior, choose data representation and inference network, generate simulations and train, validate the posterior, then analyze the posterior for scientific insight (Guide, p5:L27-L36).

## Mathematical Framework

The simulator defines a stochastic forward model

$$
x \sim p(x \mid \theta),
$$

where $\theta$ are simulator parameters and $x$ is simulated data (Guide, p2:L1-L7; p6:L2-L5).

Bayesian inference targets

$$
p(\theta \mid x_o) \propto p(x_o \mid \theta)p(\theta),
$$

where $p(\theta)$ encodes prior knowledge and $p(x_o \mid \theta)$ is the likelihood implied by the simulator (Guide, p3:L27-L35; p4:L1-L2).

SBI replaces explicit likelihood evaluation with a simulated training set

$$
\mathcal{D}=\{(\theta_i,x_i)\}_{i=1}^{N}, \qquad
\theta_i \sim p(\theta), \quad x_i \sim p(x \mid \theta_i),
$$

drawn from $p(\theta,x)=p(\theta)p(x\mid\theta)$ (Guide, p4:L3-L10).

In neural posterior estimation, the learned object is

$$
q_\phi(\theta \mid x) \approx p(\theta \mid x),
$$

and inference for a new observation is a forward evaluation $q_\phi(\theta \mid x_o)$, often amortized across many observations (Guide, p4:L29-L40; p12:L1-L3).
