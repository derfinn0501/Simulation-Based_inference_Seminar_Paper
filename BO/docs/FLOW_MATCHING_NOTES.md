# Flow Matching Notes

## Core Idea

Flow matching learns a vector field that transports samples from a simple base
distribution to a target distribution.

For posterior inference, the target distribution is conditional:

```text
p(theta | x)
```

The learned model should define a conditional vector field:

```text
v_phi(t, theta_t, x)
```

or, when design variables are included:

```text
v_phi(t, theta_t, x, psi)
```

## Simple Rectified-Flow Form

A minimal training construction samples:

```text
theta_0 ~ base noise
theta_1 ~ posterior training target
t ~ Uniform(0, 1)
theta_t = (1 - t) theta_0 + t theta_1
target_velocity = theta_1 - theta_0
```

The model learns:

```text
v_phi(t, theta_t, condition) approx theta_1 - theta_0
```

Sampling starts from base noise and integrates the learned vector field.

## Project Caution

The current rectified FMPE implementation is lightweight.
It is useful for controlled diagnostics, but it is not yet a production-quality
conditional flow-matching posterior estimator.

Current open issues:

- calibration is weaker than point-estimate performance
- exact posterior density is not available
- neural architecture and training budget are not yet tuned
- evaluation must include more than posterior plots
