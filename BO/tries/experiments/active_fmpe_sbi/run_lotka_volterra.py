#!/usr/bin/env python3
"""Active simulation design prototype for Lotka-Volterra SBI.

This is deliberately the first, simple design from the project notes:

- infer physical parameters theta from predator-prey observations,
- let BO choose design variables psi, not theta,
- reward BO by posterior-quality improvement on held-out simulator pairs,
- use a simple NPE-style Gaussian posterior baseline before replacing it with FMPE.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


THETA_BOUNDS = np.array(
    [
        [0.5, 1.5],   # alpha: prey growth
        [0.02, 0.08], # beta: predation rate
        [0.5, 1.5],   # gamma: predator death
        [0.02, 0.08], # delta: predator reproduction from prey
    ],
    dtype=float,
)


@dataclass
class Design:
    """Physical design decoded from unit-cube BO variables."""

    prey0: float
    pred0: float
    t_start: float
    t_span: float

    @property
    def t_end(self) -> float:
        return self.t_start + self.t_span


def design_dim(design_space: str) -> int:
    if design_space == "full":
        return 4
    if design_space in {"window", "hard_window", "wide_window"}:
        return 2
    raise ValueError(f"Unknown design space: {design_space}")


def decode_design(psi_unit: np.ndarray, design_space: str = "full") -> Design:
    """Map unit-cube psi to interpretable experiment design variables.

    full:
        BO controls initial populations and observation window.
    window:
        BO controls only observation window; initial populations are fixed.
    hard_window:
        BO controls only observation window over a wider, harder range. Many
        random windows become weakly informative, making BO's role easier to
        diagnose.
    wide_window:
        BO controls only observation window over a long horizon and a broad
        duration range. With fixed n_obs, this lets BO choose between short
        dense local dynamics and longer trend observations.
    """

    psi = np.clip(np.asarray(psi_unit, dtype=float), 0.0, 1.0)
    if design_space == "full":
        prey0 = 20.0 + 40.0 * psi[0]
        pred0 = 5.0 + 25.0 * psi[1]
        t_span = 4.0 + 12.0 * psi[2]
        t_start = (20.0 - t_span) * psi[3]
    elif design_space == "window":
        prey0 = 40.0
        pred0 = 15.0
        t_span = 4.0 + 12.0 * psi[0]
        t_start = (20.0 - t_span) * psi[1]
    elif design_space == "hard_window":
        prey0 = 40.0
        pred0 = 15.0
        t_span = 2.0 + 6.0 * psi[0]
        t_start = (36.0 - t_span) * psi[1]
    elif design_space == "wide_window":
        prey0 = 40.0
        pred0 = 15.0
        t_span = 2.0 + 22.0 * psi[0]
        t_start = (40.0 - t_span) * psi[1]
    else:
        raise ValueError(f"Unknown design space: {design_space}")
    return Design(prey0=prey0, pred0=pred0, t_start=t_start, t_span=t_span)


def sample_theta(rng: np.random.Generator, n: int) -> np.ndarray:
    lo = THETA_BOUNDS[:, 0]
    hi = THETA_BOUNDS[:, 1]
    return rng.uniform(lo, hi, size=(n, len(lo)))


def sample_psi(rng: np.random.Generator, n: int, design_space: str) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, design_dim(design_space)))


def lotka_rhs(_t: float, y: np.ndarray, theta: np.ndarray) -> list[float]:
    prey, pred = np.maximum(y, 1e-9)
    alpha, beta, gamma, delta = theta
    return [
        alpha * prey - beta * prey * pred,
        delta * prey * pred - gamma * pred,
    ]


def simulate_one(
    theta: np.ndarray,
    psi_unit: np.ndarray,
    rng: np.random.Generator,
    n_obs: int,
    noise_std: float,
    design_space: str,
) -> np.ndarray:
    """Simulate log-observations for one theta/design pair."""

    design = decode_design(psi_unit, design_space=design_space)
    times = np.linspace(design.t_start, design.t_end, n_obs)
    y0 = np.array([design.prey0, design.pred0], dtype=float)
    sol = solve_ivp(
        fun=lambda t, y: lotka_rhs(t, y, theta),
        t_span=(0.0, max(design.t_end, 1e-6)),
        y0=y0,
        t_eval=times,
        method="LSODA",
        rtol=1e-5,
        atol=1e-7,
    )
    if not sol.success or sol.y.shape[1] != n_obs or not np.isfinite(sol.y).all():
        return np.full(2 * n_obs, np.nan)

    populations = np.maximum(sol.y.T, 1e-9)
    obs = np.log1p(populations)
    obs = obs + rng.normal(0.0, noise_std, size=obs.shape)
    return obs.reshape(-1)


def simulate_batch(
    theta: np.ndarray,
    psi: np.ndarray,
    rng: np.random.Generator,
    n_obs: int,
    noise_std: float,
    design_space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    kept_theta: list[np.ndarray] = []
    kept_psi: list[np.ndarray] = []
    for th, ps in zip(theta, psi):
        x = simulate_one(th, ps, rng, n_obs=n_obs, noise_std=noise_std, design_space=design_space)
        if np.isfinite(x).all():
            xs.append(x)
            kept_theta.append(th)
            kept_psi.append(ps)
    if not xs:
        raise RuntimeError("All simulations failed; loosen simulator settings.")
    return np.asarray(kept_theta), np.asarray(kept_psi), np.asarray(xs)


def make_features(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return np.hstack([x, psi])


class GaussianPosteriorRegressor:
    """Simple NPE-style baseline q(theta | x, psi).

    The model predicts posterior mean with an MLP and uses a global residual
    covariance in standardized theta-space. It is not the final FMPE model, but
    gives us a scalar log-posterior quality signal for the first BO prototype.
    """

    def __init__(self, seed: int):
        self.x_scaler = StandardScaler()
        self.theta_scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(96, 96),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=2e-3,
            max_iter=650,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=seed,
        )
        self.cov: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None
        self.log_det_cov: float | None = None

    def fit(self, x: np.ndarray, psi: np.ndarray, theta: np.ndarray) -> "GaussianPosteriorRegressor":
        features = make_features(x, psi)
        x_std = self.x_scaler.fit_transform(features)
        theta_std = self.theta_scaler.fit_transform(theta)
        self.model.fit(x_std, theta_std)
        pred = self.model.predict(x_std)
        resid = theta_std - pred
        cov = np.cov(resid.T)
        cov = np.atleast_2d(cov) + np.eye(theta.shape[1]) * 0.03
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            raise RuntimeError("Residual covariance is not positive definite.")
        self.log_det_cov = float(log_det)
        return self

    def predict_mean(self, x: np.ndarray, psi: np.ndarray) -> np.ndarray:
        features = make_features(x, psi)
        pred_std = self.model.predict(self.x_scaler.transform(features))
        return self.theta_scaler.inverse_transform(pred_std)

    def log_prob(self, theta: np.ndarray, x: np.ndarray, psi: np.ndarray) -> np.ndarray:
        if self.inv_cov is None or self.log_det_cov is None:
            raise RuntimeError("Model must be fit before log_prob.")
        features = make_features(x, psi)
        pred_std = self.model.predict(self.x_scaler.transform(features))
        theta_std = self.theta_scaler.transform(theta)
        diff = theta_std - pred_std
        mahal = np.einsum("ni,ij,nj->n", diff, self.inv_cov, diff)
        dim = theta.shape[1]
        logp_std = -0.5 * (dim * math.log(2.0 * math.pi) + self.log_det_cov + mahal)
        # Change of variables from standardized theta back to physical theta.
        log_abs_det = np.sum(np.log(self.theta_scaler.scale_))
        return logp_std - log_abs_det


def validation_score(model: GaussianPosteriorRegressor, val: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    theta_val, psi_val, x_val = val
    return float(np.mean(model.log_prob(theta_val, x_val, psi_val)))


def coverage_error(
    model: GaussianPosteriorRegressor,
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> float:
    """Marginal Gaussian coverage error in standardized theta-space."""

    theta_val, psi_val, x_val = val
    pred = model.predict_mean(x_val, psi_val)
    theta_std = model.theta_scaler.transform(theta_val)
    pred_std = model.theta_scaler.transform(pred)
    std = np.sqrt(np.diag(model.cov))
    total = 0.0
    for level in levels:
        z = norm.ppf(0.5 + level / 2.0)
        inside = np.abs(theta_std - pred_std) <= z * std
        total += abs(float(np.mean(inside)) - level)
    return total / len(levels)


def train_and_score(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed: int,
) -> tuple[GaussianPosteriorRegressor, float, float]:
    theta_train, psi_train, x_train = train
    model = GaussianPosteriorRegressor(seed=seed).fit(x_train, psi_train, theta_train)
    return model, validation_score(model, val), coverage_error(model, val)


def append_data(
    base: tuple[np.ndarray, np.ndarray, np.ndarray],
    extra: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.vstack([b, e]) for b, e in zip(base, extra))  # type: ignore[return-value]


def random_design_run(
    rng: np.random.Generator,
    initial: int,
    batch: int,
    rounds: int,
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_obs: int,
    noise_std: float,
    seed: int,
    design_space: str,
    replicate: int,
) -> list[dict[str, float | int | str]]:
    train = simulate_batch(
        sample_theta(rng, initial),
        sample_psi(rng, initial, design_space),
        rng,
        n_obs,
        noise_std,
        design_space,
    )
    rows: list[dict[str, float | int | str]] = []
    for r in range(rounds + 1):
        _, score, cov_err = train_and_score(train, val, seed=seed + 1000 + r)
        rows.append(
            {
                "method": "random",
                "replicate": replicate,
                "round": r,
                "simulations": len(train[0]),
                "validation_log_posterior": score,
                "coverage_error": cov_err,
            }
        )
        if r == rounds:
            break
        extra = simulate_batch(
            sample_theta(rng, batch),
            sample_psi(rng, batch, design_space),
            rng,
            n_obs,
            noise_std,
            design_space,
        )
        train = append_data(train, extra)
    return rows


def propose_design(
    rng: np.random.Generator,
    tried_psi: list[np.ndarray],
    rewards: list[float],
    n_candidates: int,
    design_space: str,
    exploit_best_prob: float,
) -> np.ndarray:
    """BO proposal over psi in a unit-cube design space.

    Uses random exploration until enough observations exist, then a GP-UCB
    acquisition over a random candidate set.
    """

    if rewards and rng.uniform() < exploit_best_prob:
        return tried_psi[int(np.argmax(rewards))]

    if len(rewards) < 4:
        return sample_psi(rng, 1, design_space)[0]

    x_train = np.asarray(tried_psi)
    y_train = np.asarray(rewards)
    y_std = (y_train - y_train.mean()) / (y_train.std() + 1e-8)
    dim = design_dim(design_space)
    kernel = Matern(length_scale=np.ones(dim) * 0.4, nu=2.5) + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=int(rng.integers(1e9)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x_train, y_std)
    best_psi = tried_psi[int(np.argmax(rewards))]
    candidates = np.vstack([sample_psi(rng, n_candidates, design_space), best_psi[None, :], np.asarray(tried_psi)])
    mean, std = gp.predict(candidates, return_std=True)
    acquisition = mean + 1.5 * std
    return candidates[int(np.argmax(acquisition))]


def bo_design_run(
    rng: np.random.Generator,
    initial: int,
    batch: int,
    rounds: int,
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_obs: int,
    noise_std: float,
    seed: int,
    n_candidates: int,
    design_space: str,
    replicate: int,
    exploit_best_prob: float,
    bo_random_fraction: float,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int]]]:
    train = simulate_batch(
        sample_theta(rng, initial),
        sample_psi(rng, initial, design_space),
        rng,
        n_obs,
        noise_std,
        design_space,
    )
    rows: list[dict[str, float | int | str]] = []
    trace: list[dict[str, float | int]] = []
    tried_psi: list[np.ndarray] = []
    rewards: list[float] = []

    _, old_score, old_cov = train_and_score(train, val, seed=seed + 2000)
    rows.append(
            {
                "method": "bo",
                "replicate": replicate,
                "round": 0,
                "simulations": len(train[0]),
                "validation_log_posterior": old_score,
            "coverage_error": old_cov,
        }
    )

    for r in range(1, rounds + 1):
        psi_next = propose_design(
            rng,
            tried_psi,
            rewards,
            n_candidates=n_candidates,
            design_space=design_space,
            exploit_best_prob=exploit_best_prob,
        )
        n_random = int(round(batch * bo_random_fraction))
        n_focused = max(1, batch - n_random)
        psi_new = np.repeat(psi_next[None, :], n_focused, axis=0)
        if n_random > 0:
            psi_new = np.vstack([psi_new, sample_psi(rng, n_random, design_space)])
        theta_new = sample_theta(rng, len(psi_new))
        extra = simulate_batch(theta_new, psi_new, rng, n_obs, noise_std, design_space)
        candidate_train = append_data(train, extra)
        _, new_score, new_cov = train_and_score(candidate_train, val, seed=seed + 2000 + r)
        reward = new_score - old_score
        tried_psi.append(psi_next)
        rewards.append(reward)
        design = decode_design(psi_next, design_space=design_space)
        trace_row: dict[str, float | int] = {
            "replicate": replicate,
            "round": r,
            "reward_delta_log_posterior": reward,
            "prey0": design.prey0,
            "pred0": design.pred0,
            "t_start": design.t_start,
            "t_span": design.t_span,
        }
        for i, value in enumerate(psi_next):
            trace_row[f"psi{i}_unit"] = float(value)
        trace.append(trace_row)
        train = candidate_train
        old_score = new_score
        old_cov = new_cov
        rows.append(
            {
                "method": "bo",
                "replicate": replicate,
                "round": r,
                "simulations": len(train[0]),
                "validation_log_posterior": new_score,
                "coverage_error": new_cov,
            }
        )
    return rows, trace


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, float | int | str]], output: Path) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    plt.figure(figsize=(8, 5))
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in subset:
            grouped[int(row["simulations"])].append(float(row["validation_log_posterior"]))
        xs = sorted(grouped)
        means = np.array([np.mean(grouped[x]) for x in xs])
        stds = np.array([np.std(grouped[x]) for x in xs])
        plt.plot(xs, means, marker="o", label=method)
        if any(len(grouped[x]) > 1 for x in xs):
            plt.fill_between(xs, means - stds, means + stds, alpha=0.18)
    plt.xlabel("Simulator calls")
    plt.ylabel("Validation log posterior (higher is better)")
    plt.title("Lotka-Volterra active simulation design")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=int, default=140)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--validation", type=int, default=240)
    parser.add_argument("--n-obs", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.06)
    parser.add_argument("--bo-candidates", type=int, default=256)
    parser.add_argument("--exploit-best-prob", type=float, default=0.25)
    parser.add_argument("--bo-random-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--design-space",
        choices=("full", "window", "hard_window", "wide_window"),
        default="full",
        help="Design variables BO/random control. Use wide_window to test short-vs-long observation windows.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--quick", action="store_true", help="Use a tiny run for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.initial = 60
        args.batch = 10
        args.rounds = 2
        args.validation = 80
        args.bo_candidates = 64
        args.repeats = 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    bo_trace: list[dict[str, float | int | str]] = []
    for rep in range(args.repeats):
        rep_seed = args.seed + 10_000 * rep
        rng = np.random.default_rng(rep_seed)
        val = simulate_batch(
            sample_theta(rng, args.validation),
            sample_psi(rng, args.validation, args.design_space),
            rng,
            args.n_obs,
            args.noise_std,
            args.design_space,
        )

        random_rows = random_design_run(
            rng=np.random.default_rng(rep_seed + 1),
            initial=args.initial,
            batch=args.batch,
            rounds=args.rounds,
            val=val,
            n_obs=args.n_obs,
            noise_std=args.noise_std,
            seed=rep_seed,
            design_space=args.design_space,
            replicate=rep,
        )
        bo_rows, trace = bo_design_run(
            rng=np.random.default_rng(rep_seed + 2),
            initial=args.initial,
            batch=args.batch,
            rounds=args.rounds,
            val=val,
            n_obs=args.n_obs,
            noise_std=args.noise_std,
            seed=rep_seed,
            n_candidates=args.bo_candidates,
            design_space=args.design_space,
            replicate=rep,
            exploit_best_prob=args.exploit_best_prob,
            bo_random_fraction=args.bo_random_fraction,
        )
        rows.extend(random_rows + bo_rows)
        bo_trace.extend(trace)

    write_csv(args.output_dir / "metrics.csv", rows)
    write_csv(args.output_dir / "bo_trace.csv", bo_trace)
    plot_metrics(rows, args.output_dir / "posterior_quality.png")

    print(f"Wrote metrics to {args.output_dir / 'metrics.csv'}")
    print(f"Wrote BO trace to {args.output_dir / 'bo_trace.csv'}")
    print(f"Wrote plot to {args.output_dir / 'posterior_quality.png'}")
    print("Final rows:")
    for row in rows:
        if int(row["round"]) == args.rounds:
            print(row)


if __name__ == "__main__":
    main()
