# FMPE Introductory Idea

## Intuition

Flow-matching posterior estimation (FMPE) is a neural posterior estimation method for SBI. Like NPE, it learns a conditional posterior estimator $q_\phi(\theta \mid x)$ from simulated pairs $(\theta,x)$, but it replaces discrete normalizing flows with a continuous flow defined by a learned vector field (FMPE, p1:L29-L37; FMPE, p4:L35-L43).

The core picture is: start with simple noise in parameter space, then learn a velocity field that moves this noise continuously into samples from the posterior. Instead of learning a sequence of invertible discrete transformations, FMPE learns a time-dependent vector map that tells each point where to move next (FMPE, p2:L115-L118; FMPE, p3:L47-L53).

This is useful for SBI because the observation $x$ can be high-dimensional and complex, while the inferred parameters $\theta$ are often much lower-dimensional. FMPE can spend more architecture capacity on interpreting $x$ and less on satisfying the architectural constraints of discrete normalizing flows (FMPE, p6:L29-L39; FMPE, p9:L164-L173).

FMPE remains an amortized posterior method: after training, the learned model can infer posteriors for new observations. It is therefore not a replacement for the SBI workflow in the practical guide, but a more flexible choice for the posterior-estimation network inside that workflow (PG, p5:L27-L36; FMPE, p5:L17-L22).

## Mathematical Framework

The target remains the Bayesian posterior

$$
p(\theta \mid x),
$$

where $\theta$ are physical or simulator parameters and $x$ is observed or simulated data (FMPE, p2:L88-L97).

As in NPE, training data are generated from the simulator:

$$
\theta \sim p(\theta),
\qquad
x \sim p(x \mid \theta).
$$

The FMPE paper emphasizes that this is the same empirical training distribution used in NPE: samples $(\theta,x)$ are drawn from $p(\theta)p(x\mid\theta)$ (FMPE, p4:L35-L43; FMPE, p5:L17-L22).

FMPE defines a continuous flow in parameter space. Let $t\in[0,1]$ be artificial flow time and let $\psi_{t,x}$ map a base sample through time. The sample trajectory obeys an ODE:

$$
\frac{d}{dt}\psi_{t,x}(\theta)
=
v_{t,x}(\psi_{t,x}(\theta)),
\qquad
\psi_{0,x}(\theta)=\theta.
$$

Here $v_{t,x}$ is the learned vector field conditioned on the observation $x$ (FMPE, p3:L47-L53).

Flow matching trains this vector field by regression. The FMPE objective is

$$
\mathcal{L}_{\text{FMPE}}
=
\mathbb{E}
\left[
\left\|
v_{t,x}(\theta_t)
-
u_t(\theta_t \mid \theta_1)
\right\|^2
\right],
$$

with

$$
t\sim p(t),
\qquad
\theta_1\sim p(\theta),
\qquad
x\sim p(x\mid\theta_1),
\qquad
\theta_t\sim p_t(\theta_t\mid\theta_1).
$$

The target vector field $u_t$ is known from the chosen probability path, so training does not require solving the ODE during every optimization step (FMPE, p4:L12-L20; FMPE, p5:L10-L22).

After training, posterior samples are produced by starting from the base distribution and integrating the learned vector field from $t=0$ to $t=1$:

$$
\theta_0 \sim q_0(\theta),
\qquad
\frac{d\theta_t}{dt}
=
v_{t,x}(\theta_t),
\qquad
\theta_1 \sim q_\phi(\theta\mid x).
$$

This is the central object of FMPE: posterior inference through a learned vector map.
