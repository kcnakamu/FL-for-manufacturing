"""
Aggregate competence matrices across training seeds.

Answers one question: which per-class teacher rankings survive seed noise?

A single competence matrix gives a margin between the best and second-best
teacher for each class, but no scale to judge it against. Re-running the whole
bank under different training seeds supplies that scale. Two signals are
reported per class, and they disagree in informative ways:

  argmax stability -- does the same teacher win this class in every seed? With
      few seeds this is the more trustworthy signal, because it makes no
      distributional assumption. A flipping argmax means the ranking is noise
      regardless of what the margin looks like.

  margin vs sigma  -- the mean top1-top2 gap divided by the standard error of
      that gap. Treated as decisive only past ~2. With n=3 the sigma estimate
      is itself rough, so this is a guide, not a test.

A class is called DECISIVE only when both agree: one teacher wins every seed
AND the margin clears 2 sigma. Anything else is reported as noise, which is a
finding rather than a failure -- it says those teachers are interchangeable for
that class.

Usage:
    python scripts/aggregate_competence.py \
        experiments/teacher_bank_seed*/competence/competence_matrix.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

TEACHERS = [f"local_c{i}" for i in range(1, 7)]
DECISIVE_SIGMA = 2.0
# Below this, no teacher has meaningfully learned the class, so "which teacher is
# best" is not a question worth answering. Without this floor a column where every
# teacher scores 0.0 yields margin 0 over sigma 0 -> inf -> a false DECISIVE.
MIN_USEFUL_AP = 0.01


def load(paths: list[Path]) -> tuple[list[str], dict]:
    runs, class_names = [], None
    for p in paths:
        d = json.loads(p.read_text())
        if class_names is None:
            class_names = d["class_names"]
        elif d["class_names"] != class_names:
            raise ValueError(f"{p}: class_names differ from the first run; cannot aggregate")
        seed = d.get("seed")
        if seed is None:
            raise ValueError(
                f"{p}: no seed recorded. Regenerate with the current "
                f"competence_matrix.py so runs can be told apart."
            )
        runs.append({"seed": seed, "path": p, "matrix": d["matrix"]})

    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds {sorted(seeds)} -- each run must be a distinct seed")
    runs.sort(key=lambda r: r["seed"])
    return class_names, runs


def aggregate(class_names: list[str], runs: list[dict]) -> tuple[dict, list[dict]]:
    n = len(runs)

    cells = {
        t: {
            c: {
                "mean": st.mean(vals := [r["matrix"][t][c] for r in runs]),
                "std": st.stdev(vals) if n > 1 else 0.0,
                "values": vals,
            }
            for c in class_names
        }
        for t in TEACHERS
    }

    per_class = []
    for c in class_names:
        # Rank within each seed, then ask how often the same teacher wins.
        winners = [max(TEACHERS, key=lambda t: r["matrix"][t][c]) for r in runs]
        modal, n_modal = Counter(winners).most_common(1)[0]
        stable = n_modal == n

        # Margin measured per seed, so its spread reflects the gap's own noise.
        margins = []
        for r in runs:
            ordered = sorted((r["matrix"][t][c] for t in TEACHERS), reverse=True)
            margins.append(ordered[0] - ordered[1])
        m_mean = st.mean(margins)
        m_std = st.stdev(margins) if n > 1 else 0.0

        # Standard error of the top1-top2 difference, from the two cells involved.
        runner = sorted(TEACHERS, key=lambda t: -cells[t][c]["mean"])[1]
        s1, s2 = cells[modal][c]["std"], cells[runner][c]["std"]
        sigma_diff = (s1**2 + s2**2) ** 0.5
        if sigma_diff > 0:
            ratio = m_mean / sigma_diff
        else:
            # Zero variance: separated only if there is a gap to separate.
            ratio = float("inf") if m_mean > 0 else 0.0

        if cells[modal][c]["mean"] < MIN_USEFUL_AP:
            verdict = "DEGENERATE"   # nobody learned this class
        elif m_mean <= 0:
            verdict = "NOISE"
        elif stable and ratio >= DECISIVE_SIGMA:
            verdict = "DECISIVE"
        else:
            verdict = "NOISE"

        per_class.append({
            "class": c,
            "argmax": modal,
            "argmax_stable": stable,
            "argmax_agreement": f"{n_modal}/{n}",
            "winners_by_seed": winners,
            "top1_mean": cells[modal][c]["mean"],
            "top1_std": cells[modal][c]["std"],
            "runner_up": runner,
            "runner_up_mean": cells[runner][c]["mean"],
            "margin_mean": m_mean,
            "margin_std": m_std,
            "sigma_diff": sigma_diff,
            "margin_over_sigma": ratio,
            "verdict": verdict,
        })
    return cells, per_class


def report(class_names: list[str], runs: list[dict], cells: dict, per_class: list[dict]) -> None:
    n = len(runs)
    seeds = [r["seed"] for r in runs]
    w, rw = 18, 30

    print("\n" + "=" * 76)
    print(f"COMPETENCE MATRIX across {n} seeds {seeds} -- mean +/- std of per-class mAP50")
    print("=" * 76)
    head = f"{'':<{rw}s}" + "".join(f"{c[:16]:>{w}s}" for c in class_names)
    print(head); print("-" * len(head))
    for i, t in enumerate(TEACHERS, 1):
        row = f"{t + '  (C' + str(i) + ')':<{rw}s}"
        for c in class_names:
            row += f"{cells[t][c]['mean']:.3f}+/-{cells[t][c]['std']:.3f}".rjust(w)
        print(row)

    print("\n" + "=" * 76)
    print("PER-CLASS VERDICT")
    print("=" * 76)
    print(f"{'class':<18s}{'argmax':<12s}{'stable':<9s}{'top1':<16s}"
          f"{'margin':<16s}{'m/sigma':<9s}{'verdict'}")
    print("-" * 90)
    for s in per_class:
        print(
            f"{s['class']:<18s}{s['argmax']:<12s}{s['argmax_agreement']:<9s}"
            f"{s['top1_mean']:.3f}+/-{s['top1_std']:.3f}   "
            f"{s['margin_mean']:.4f}+/-{s['margin_std']:.4f}  "
            f"{s['margin_over_sigma']:>7.2f}  {s['verdict']}"
        )

    dec = [s for s in per_class if s["verdict"] == "DECISIVE"]
    deg = [s for s in per_class if s["verdict"] == "DEGENERATE"]
    print(f"\n{len(dec)}/{len(per_class)} classes have a teacher that survives seed noise: "
          f"{', '.join(s['class'] for s in dec) or '(none)'}")
    if deg:
        print(f"{len(deg)} class(es) degenerate -- no teacher exceeds "
              f"{MIN_USEFUL_AP} mAP50, so there is no teacher to choose: "
              f"{', '.join(s['class'] for s in deg)}")
    for s in per_class:
        if not s["argmax_stable"]:
            print(f"  ! {s['class']}: argmax flips across seeds {s['winners_by_seed']}")
    if n < 3:
        print(f"\n[WARN] only {n} seed(s): std is not meaningful. Use 3 minimum, 5 preferred.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate competence matrices across seeds.")
    ap.add_argument("runs", nargs="+", help="competence_matrix.json files, one per seed")
    ap.add_argument("--out_dir", default="experiments/competence_across_seeds")
    args = ap.parse_args()

    paths = [Path(p) for p in args.runs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")

    class_names, runs = load(paths)
    cells, per_class = aggregate(class_names, runs)
    report(class_names, runs, cells, per_class)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "competence_across_seeds.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["teacher"] + [f"{c}_{k}" for c in class_names for k in ("mean", "std")])
        for t in TEACHERS:
            row = [t]
            for c in class_names:
                row += [f"{cells[t][c]['mean']:.6f}", f"{cells[t][c]['std']:.6f}"]
            wr.writerow(row)
    (out / "competence_across_seeds.json").write_text(json.dumps({
        "seeds": [r["seed"] for r in runs],
        "sources": [str(r["path"]) for r in runs],
        "class_names": class_names,
        "cells": cells,
        "per_class": per_class,
    }, indent=2))
    print(f"\n[DONE] -> {out}/competence_across_seeds.{{csv,json}}")


if __name__ == "__main__":
    main()
