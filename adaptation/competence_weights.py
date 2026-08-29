"""Turn a measured competence matrix into KD weights.

Consumes competence_across_seeds.json (scripts/aggregate_competence.py) and
produces the two quantities multi-teacher distillation needs. They answer
different questions and must not be conflated:

    lambda_c  (nc,)      how much to distill class c AT ALL.
                         Driven by the best competence any teacher has for c:
                         if nobody knows a class, distilling it just injects
                         noise into the student.

    w[k][c]   (K, nc)    WHICH teacher to believe for class c, normalized over
                         teachers so each class column is a convex combination.
                         This is what makes p_ens = sum_k w[k,c]*sigmoid(z_k/T)
                         a valid probability.

Teacher weights use a variance-scaled softmax:

    w[k,c]  proportional to  exp( mu[k,c] / (tau * sigma_bar[c]) )

Dividing by the class's own noise scale is the point. The measured matrix has
seed-to-seed sigma of 0.03-0.07 on contested cells while the gaps between top
teachers are 0.01-0.05, so an unscaled softmax would read those gaps as real.
Scaling by sigma_bar flattens the weights exactly where teachers are
statistically indistinguishable and sharpens them where they are not -- which
is the whole reason argmax routing was unstable across seeds.

tau is the one knob, and the ablations are its limits:
    tau -> 0     one-hot: hard argmax routing (ties split evenly)
    tau -> inf   uniform: FedDF-style unweighted ensembling
Both are computed directly rather than by pushing exp() to its limits.

Pure Python -- no torch, no Ultralytics -- so it is CPU-testable in isolation
(adaptation/tests/test_competence_weights.py).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

# Guards the variance-scaled softmax when a class has zero measured spread
# (every teacher identically 0.000, which happens for a class nobody detects).
SIGMA_FLOOR = 1e-3


@dataclass
class CompetenceWeights:
    class_names: list[str]
    teachers: list[str]
    tau: float
    seeds: list[int]
    mu: dict[str, dict[str, float]]
    sigma: dict[str, dict[str, float]]
    lambda_c: list[float]          # (nc,) sums to 1, ordered by class index
    w: list[list[float]]           # w[k][c]; every class column sums to 1

    def effective_teachers(self, c_idx: int) -> float:
        """1 / sum_k w^2 -- ~1.0 means monopoly routing, ~K means uniform."""
        s = sum(self.w[k][c_idx] ** 2 for k in range(len(self.teachers)))
        return 1.0 / s if s > 0 else float(len(self.teachers))

    def to_dict(self) -> dict:
        return {
            "tau": self.tau,
            "seeds": self.seeds,
            "class_names": self.class_names,
            "teachers": self.teachers,
            "lambda_c": {c: self.lambda_c[i] for i, c in enumerate(self.class_names)},
            "w": {t: {c: self.w[k][i] for i, c in enumerate(self.class_names)}
                  for k, t in enumerate(self.teachers)},
            "effective_teachers": {c: self.effective_teachers(i)
                                   for i, c in enumerate(self.class_names)},
        }


def load_competence(path: str | Path, allow_single_seed: bool = False) -> dict:
    """Read competence_across_seeds.json, validating what the weights depend on."""
    path = Path(path)
    d = json.loads(path.read_text())

    for key in ("class_names", "cells", "seeds"):
        if key not in d:
            raise ValueError(
                f"{path}: missing '{key}'. Expected the output of "
                f"scripts/aggregate_competence.py, not a single-run competence_matrix.json."
            )

    seeds = d["seeds"]
    if len(seeds) < 2 and not allow_single_seed:
        raise ValueError(
            f"{path}: only {len(seeds)} seed(s). The variance scaling needs a real "
            f"sigma; a single run would bake that seed's coin flips into the weights. "
            f"Aggregate >=3 seeds, or pass allow_single_seed to override."
        )
    return d


def _column_weights(mu_col: list[float], sigma_col: list[float], tau: float) -> list[float]:
    """Weights over teachers for ONE class. Returns a list summing to 1."""
    k = len(mu_col)

    if tau == float("inf"):
        return [1.0 / k] * k

    if tau == 0.0:
        best = max(mu_col)
        winners = [i for i, v in enumerate(mu_col) if v == best]
        return [1.0 / len(winners) if i in winners else 0.0 for i in range(k)]

    if tau < 0:
        raise ValueError(f"tau must be >= 0 (or inf), got {tau}")

    sigma_bar = max(sum(sigma_col) / k, SIGMA_FLOOR)
    scaled = [m / (tau * sigma_bar) for m in mu_col]
    # Shift by the max before exp: mu/(tau*sigma) reaches ~40 on this data, and
    # exp(40) is fine but exp of a larger future value would not be.
    hi = max(scaled)
    exps = [math.exp(v - hi) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def derive_weights(
    competence: dict,
    tau: float = 3.0,
    min_competence: float = 0.0,
) -> CompetenceWeights:
    """Build (lambda_c, w[k][c]) from an aggregated competence matrix.

    Args:
        competence: parsed competence_across_seeds.json.
        tau: softmax temperature in units of the class's own sigma. 0 = argmax
            routing, inf = uniform ensembling.
        min_competence: classes whose best teacher scores below this get
            lambda_c = 0 -- nobody knows them, so distilling them adds noise.
    """
    class_names = competence["class_names"]
    cells = competence["cells"]
    teachers = sorted(cells)          # local_c1..local_c6

    mu = {t: {c: float(cells[t][c]["mean"]) for c in class_names} for t in teachers}
    sigma = {t: {c: float(cells[t][c]["std"]) for c in class_names} for t in teachers}

    # lambda_c: driven by the best available teacher for each class.
    best = [max(mu[t][c] for t in teachers) for c in class_names]
    gated = [0.0 if b < min_competence else b for b in best]
    total = sum(gated)
    if total <= 0:
        raise ValueError(
            f"No class has a teacher above min_competence={min_competence}; "
            f"best per class was {dict(zip(class_names, best))}."
        )
    lambda_c = [g / total for g in gated]

    # w[k][c]: one softmax per class column, over teachers.
    w = [[0.0] * len(class_names) for _ in teachers]
    for i, c in enumerate(class_names):
        col = _column_weights([mu[t][c] for t in teachers],
                              [sigma[t][c] for t in teachers], tau)
        for k in range(len(teachers)):
            w[k][i] = col[k]

    return CompetenceWeights(
        class_names=class_names, teachers=teachers, tau=tau,
        seeds=competence["seeds"], mu=mu, sigma=sigma,
        lambda_c=lambda_c, w=w,
    )


def check_class_order(cw: CompetenceWeights, data_yaml: str | Path) -> None:
    """Fail loudly if the weight ordering does not match the dataset's class order.

    lambda_c and w are indexed by class INDEX when they reach the loss, so a
    mismatch here silently applies each class's weights to a different channel.
    That is the same misalignment failure mode the competence matrix itself had
    to be defended against, and it is invisible in the numbers downstream.
    """
    import yaml

    names = [str(n) for n in yaml.safe_load(Path(data_yaml).read_text())["names"]]
    if names != cw.class_names:
        raise ValueError(
            f"Class order mismatch.\n  weights : {cw.class_names}\n  "
            f"{data_yaml}: {names}\nWeights are applied by class index, so these "
            f"must match element-for-element."
        )


def load_kd_weights(path: str | Path, class_names: list[str],
                    teachers: list[str] | None = None):
    """Read a kd_weights JSON back into (lambda_c, w[k][c]) for KDDetectionLoss.

    Both are consumed positionally by class index and teacher index, so the
    ordering is validated rather than trusted: `class_names` must be the dataset's
    order and `teachers` the order the checkpoints are loaded in. A silent
    permutation here would apply every class's weights to the wrong logit
    channel, and nothing downstream would look wrong.

    Returns:
        (lambda_c, w) -- lambda_c is (nc,), w is (K, nc) as nested lists.
    """
    d = json.loads(Path(path).read_text())
    file_classes = d["class_names"]
    if file_classes != list(class_names):
        raise ValueError(
            f"class order mismatch.\n  {path}: {file_classes}\n  dataset: "
            f"{list(class_names)}\nWeights are applied by class index."
        )

    file_teachers = d["teachers"]
    teachers = list(teachers) if teachers is not None else file_teachers
    missing = [t for t in teachers if t not in d["w"]]
    if missing:
        raise ValueError(f"{path} has no weights for teacher(s) {missing} "
                         f"(has {sorted(d['w'])}).")

    lambda_c = [float(d["lambda_c"][c]) for c in class_names]
    w = [[float(d["w"][t][c]) for c in class_names] for t in teachers]

    for i, c in enumerate(class_names):
        col = sum(w[k][i] for k in range(len(teachers)))
        if abs(col - 1.0) > 1e-4:
            raise ValueError(
                f"teacher weights for '{c}' sum to {col:.6f}, not 1. If you "
                f"selected a subset of teachers, the columns must be renormalised "
                f"or the fused target is not a valid probability."
            )
    return lambda_c, w


def report(cw: CompetenceWeights) -> None:
    nc, K = len(cw.class_names), len(cw.teachers)
    w_col = max(max(len(c) for c in cw.class_names), 9) + 2

    print(f"\n{'=' * 78}")
    print(f"KD WEIGHTS   tau={cw.tau}   seeds={cw.seeds}")
    print("=" * 78)

    print("\nlambda_c -- how much to distill each class (best teacher's mAP50 -> share)")
    for i, c in enumerate(cw.class_names):
        best_t = max(cw.teachers, key=lambda t: cw.mu[t][c])
        print(f"  {c:<18s} best={cw.mu[best_t][c]:.3f} ({best_t})"
              f"   lambda={cw.lambda_c[i]:.4f}")

    print(f"\nw[k][c] -- which teacher to believe per class (each column sums to 1)")
    head = f"{'':<12s}" + "".join(f"{c[:w_col - 2]:>{w_col}s}" for c in cw.class_names)
    print(head); print("-" * len(head))
    for k, t in enumerate(cw.teachers):
        print(f"{t:<12s}" + "".join(f"{cw.w[k][i]:>{w_col}.4f}" for i in range(nc)))
    print("-" * len(head))
    print(f"{'SUM':<12s}" + "".join(
        f"{sum(cw.w[k][i] for k in range(K)):>{w_col}.4f}" for i in range(nc)))
    print(f"{'eff.teachers':<12s}" + "".join(
        f"{cw.effective_teachers(i):>{w_col}.2f}" for i in range(nc)))
    print("\n  eff.teachers = 1/sum(w^2): ~1 means one teacher owns the class,"
          f" ~{K} means all are equal.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive KD weights from a competence matrix.")
    ap.add_argument("competence", nargs="?",
                    default="experiments/competence_across_seeds/competence_across_seeds.json")
    ap.add_argument("--tau", type=float, default=3.0,
                    help="0 = argmax routing, inf = uniform (default: %(default)s)")
    ap.add_argument("--min_competence", type=float, default=0.0,
                    help="Zero lambda_c for classes whose best teacher is below this")
    ap.add_argument("--data_yaml", default=None,
                    help="Verify class order against this dataset yaml")
    ap.add_argument("--out", default=None, help="Write the weights as JSON")
    ap.add_argument("--allow_single_seed", action="store_true")
    args = ap.parse_args()

    comp = load_competence(args.competence, allow_single_seed=args.allow_single_seed)
    cw = derive_weights(comp, tau=args.tau, min_competence=args.min_competence)
    if args.data_yaml:
        check_class_order(cw, args.data_yaml)
        print(f"[OK] class order matches {args.data_yaml}")
    report(cw)
    if args.out:
        Path(args.out).write_text(json.dumps(cw.to_dict(), indent=2))
        print(f"\n[DONE] -> {args.out}")


if __name__ == "__main__":
    main()
