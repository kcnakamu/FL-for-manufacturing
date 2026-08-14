"""Disruption-timing analysis: when should the disruption round t* be?

The model converges quickly (mAP50 plateaus within a few rounds at 2 local
epochs), so t* and the local-epoch count E should be chosen from evidence, not
convention. Two signals, both computed retrain-free from the npz checkpoints
that shapley/logger.py already records during every FL run:

  1. Global convergence: evaluate each round's aggregated global on the shared
     test set -> mAP50 curve -> plateau round via `convergence_round`.
  2. Contribution stabilization (--per_round_shapley): at each round t,
     reconstruct all 8 coalitions from the logged client updates and compute
     exact Shapley phi_i(t) -> disruption is meaningful once contributions
     have stabilized and are clearly nonzero.

Compare several FL runs (different E, different seeds) in one call:

    python -m shapley.convergence \
        --log_dirs experiments/conv_E1_seed0/fl/shapley_logs \
                   experiments/conv_E2_seed0/fl/shapley_logs \
        --labels E1 E2 --local_epochs 1 2 \
        --test_dir data/neu_data/test \
        --out_dir experiments/convergence_analysis \
        --per_round_shapley

Outputs (under --out_dir): convergence.json, convergence.csv,
convergence_curves.png, and per-run shapley_by_round_<label>.csv/png.

The pure plateau math (`convergence_round`, `recommend_t_star`) is unit-tested
without YOLO in shapley/tests/test_convergence.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapley.logger import load_global, load_manifest, load_round
from shapley.persistence import _letters, _save_pt, coalition_label
from shapley.reconstruct import reconstruct
from shapley.shapley import coalitions, exact_shapley


# ======================================================================== #
# Pure plateau math (no YOLO / torch) -- unit tested.
# ======================================================================== #
def convergence_round(rounds: List[int], values: List[float],
                      min_delta: float = 0.005, patience: int = 2) -> Optional[int]:
    """First round after which no improvement > min_delta for `patience` rounds.

    An "improvement" means beating the best value seen so far by more than
    min_delta. Returns the round id of the last improvement once `patience`
    consecutive non-improving rounds have followed it, or None if the curve
    never exhibits such a plateau (still improving at the end).
    """
    if len(rounds) != len(values):
        raise ValueError(f"rounds ({len(rounds)}) and values ({len(values)}) differ in length.")
    if not rounds:
        return None
    best = values[0]
    best_round = rounds[0]
    stall = 0
    for r, v in zip(rounds[1:], values[1:]):
        if v > best + min_delta:
            best, best_round, stall = v, r, 0
        else:
            stall += 1
            if stall >= patience:
                return best_round
    return None


def recommend_t_star(rounds: List[int], values: List[float],
                     min_delta: float = 0.005, patience: int = 2) -> dict:
    """Summarize the curve into a t* recommendation for the disruption round.

    t* = the plateau round (training past it buys < min_delta mAP50). If the
    curve never plateaus, recommend the last round and flag it.
    """
    conv = convergence_round(rounds, values, min_delta=min_delta, patience=patience)
    by_round = dict(zip(rounds, values))
    if conv is None:
        last = rounds[-1] if rounds else None
        return {"converged": False, "convergence_round": None,
                "t_star": last,
                "mAP50_at_t_star": by_round.get(last),
                "note": "no plateau detected -- still improving; consider more rounds"}
    return {"converged": True, "convergence_round": conv, "t_star": conv,
            "mAP50_at_t_star": by_round[conv], "note": ""}


# ======================================================================== #
# YOLO-touching orchestration.
# ======================================================================== #
def available_global_rounds(log_dir) -> List[int]:
    """Sorted rounds with a logged aggregated global (round 0 = pre-FL baseline)."""
    globals_dir = Path(log_dir) / "globals"
    return sorted(int(p.stem.split("_")[-1]) for p in globals_dir.glob("global_round_*.npz"))


def round_curve(log_dir, test_yaml: str, out_dir: Path, label: str,
                num_classes: int = 3, device: Optional[str] = None,
                imgsz: int = 480) -> List[dict]:
    """Evaluate every logged global on the shared test set.

    Returns [{round, mAP50, mAP50-95, per_class_ap50}, ...] sorted by round.
    """
    from shapley.evaluate import evaluate_checkpoint

    curve = []
    for r in available_global_rounds(log_dir):
        pt = _save_pt(load_global(log_dir, r), num_classes,
                      out_dir / "globals_pt" / f"{label}_round_{r:02d}.pt")
        m = evaluate_checkpoint(pt, test_yaml, device=device, imgsz=imgsz,
                                out_dir=str(out_dir / "eval"), name=f"{label}_round_{r:02d}")
        curve.append({"round": r, "mAP50": m["mAP50"], "mAP50-95": m["mAP50-95"],
                      "per_class_ap50": m["per_class_ap50"]})
        print(f"[convergence] {label} round={r:02d}  mAP50={m['mAP50']:.4f}")
    return curve


def per_round_shapley(log_dir, test_yaml: str, out_dir: Path, label: str,
                      num_classes: int = 3, device: Optional[str] = None,
                      imgsz: int = 480, rule: Optional[str] = None) -> Dict[int, Dict[str, float]]:
    """Exact Shapley phi_i(t) at every logged round (8 coalition evals per round).

    Baseline v(emptyset) at round t is global_round_{t-1} -- the global that was
    broadcast INTO round t (same convention as persistence.run).
    """
    from shapley.evaluate import evaluate_checkpoint

    manifest = load_manifest(log_dir)
    rule = rule or manifest.get("rule", "fedavg")
    rounds = [r for r in available_global_rounds(log_dir) if r >= 1]

    phi_by_round: Dict[int, Dict[str, float]] = {}
    for t in rounds:
        updates, counts = load_round(log_dir, t)
        baseline = load_global(log_dir, t - 1)
        players = sorted(updates.keys())
        util = {}
        for S in coalitions(players):
            lab = coalition_label(S, players)
            pt = _save_pt(reconstruct(updates, counts, S, baseline=baseline, rule=rule),
                          num_classes,
                          out_dir / "coalitions_pt" / f"{label}_round_{t:02d}_{lab}.pt")
            m = evaluate_checkpoint(pt, test_yaml, device=device, imgsz=imgsz,
                                    out_dir=str(out_dir / "eval"),
                                    name=f"{label}_round_{t:02d}_{lab}")
            util[S] = m["mAP50"]
        phi_by_round[t] = exact_shapley(util, players)
        print(f"[convergence] {label} round={t:02d}  phi={phi_by_round[t]}")
    return phi_by_round


def run(
    log_dirs: List[str],
    labels: Optional[List[str]],
    test_dir: str,
    out_dir: str,
    local_epochs: Optional[List[int]] = None,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None,
    min_delta: float = 0.005,
    patience: int = 2,
    do_per_round_shapley: bool = False,
    device: Optional[str] = None,
    imgsz: int = 480,
) -> dict:
    from shapley.evaluate import build_test_yaml

    class_names = class_names or ["Inclusion", "Patches", "Scratches"]
    labels = labels or [f"run{i}" for i in range(len(log_dirs))]
    if len(labels) != len(log_dirs):
        raise ValueError("--labels must match --log_dirs in length.")
    if local_epochs is not None and len(local_epochs) != len(log_dirs):
        raise ValueError("--local_epochs must match --log_dirs in length.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_yaml = build_test_yaml(test_dir, class_names)

    runs: List[dict] = []
    try:
        for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
            curve = round_curve(log_dir, test_yaml, out, label,
                                num_classes=num_classes, device=device, imgsz=imgsz)
            rounds = [c["round"] for c in curve]
            values = [c["mAP50"] for c in curve]
            rec = recommend_t_star(rounds, values, min_delta=min_delta, patience=patience)

            entry = {"label": label, "log_dir": str(log_dir),
                     "local_epochs": local_epochs[i] if local_epochs else None,
                     "curve": curve, "recommendation": rec}
            if do_per_round_shapley:
                phi = per_round_shapley(log_dir, test_yaml, out, label,
                                        num_classes=num_classes, device=device, imgsz=imgsz)
                entry["phi_by_round"] = {str(t): v for t, v in phi.items()}
                _write_shapley_by_round(out, label, phi)
            runs.append(entry)
    finally:
        Path(test_yaml).unlink(missing_ok=True)

    results = {"config": {"min_delta": min_delta, "patience": patience,
                          "test_dir": str(test_dir), "class_names": class_names,
                          "imgsz": imgsz},
               "runs": runs}
    _write_outputs(out, results, class_names)

    print("\n[convergence] recommendations:")
    for r in runs:
        rec = r["recommendation"]
        e = f" (E={r['local_epochs']})" if r["local_epochs"] else ""
        print(f"  {r['label']}{e}: t*={rec['t_star']}  "
              f"mAP50={rec['mAP50_at_t_star']:.4f}  "
              f"{'converged' if rec['converged'] else rec['note']}")
    print(f"[convergence] done -> {out}")
    return results


def _write_shapley_by_round(out: Path, label: str, phi_by_round: Dict[int, Dict[str, float]]) -> None:
    if not phi_by_round:
        return
    players = sorted(next(iter(phi_by_round.values())).keys())
    letters = _letters(players)
    with open(out / f"shapley_by_round_{label}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round"] + [f"phi_{letters[p]}({p})" for p in players])
        for t in sorted(phi_by_round):
            w.writerow([t] + [phi_by_round[t][p] for p in players])
    _plot_shapley_by_round(out, label, phi_by_round, players, letters)


def _write_outputs(out: Path, results: dict, class_names: List[str]) -> None:
    with open(out / "convergence.json", "w") as fh:
        json.dump(results, fh, indent=2)

    with open(out / "convergence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "local_epochs", "round", "mAP50", "mAP50-95"]
                   + [f"AP50_{c}" for c in class_names])
        for r in results["runs"]:
            for c in r["curve"]:
                w.writerow([r["label"], r["local_epochs"], c["round"],
                            c["mAP50"], c["mAP50-95"]]
                           + [c["per_class_ap50"].get(cls, "") for cls in class_names])

    _plot_curves(out, results)


def _plot_curves(out: Path, results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[convergence] skipping plot (matplotlib unavailable: {e})")
        return

    runs = results["runs"]
    have_epochs = all(r["local_epochs"] for r in runs)
    fig, axes = plt.subplots(1, 2 if have_epochs else 1,
                             figsize=(12 if have_epochs else 7, 5), squeeze=False)
    ax = axes[0][0]
    for r in runs:
        rounds = [c["round"] for c in r["curve"]]
        vals = [c["mAP50"] for c in r["curve"]]
        line, = ax.plot(rounds, vals, marker="o", label=r["label"])
        t_star = r["recommendation"]["t_star"]
        if t_star is not None:
            ax.axvline(t_star, color=line.get_color(), ls=":", lw=0.8)
    ax.set_xlabel("federated round")
    ax.set_ylabel("central-test mAP50")
    ax.set_title("Global-model convergence (t* marked)")
    ax.legend()

    if have_epochs:
        ax2 = axes[0][1]
        for r in runs:
            E = r["local_epochs"]
            cum = [c["round"] * E for c in r["curve"]]
            vals = [c["mAP50"] for c in r["curve"]]
            ax2.plot(cum, vals, marker="o", label=f"{r['label']} (E={E})")
        ax2.set_xlabel("cumulative local epochs (round x E)")
        ax2.set_ylabel("central-test mAP50")
        ax2.set_title("Compute-fair comparison")
        ax2.legend()

    fig.tight_layout()
    fig.savefig(out / "convergence_curves.png", dpi=150)
    plt.close(fig)
    print(f"[convergence] wrote {out / 'convergence_curves.png'}")


def _plot_shapley_by_round(out: Path, label: str, phi_by_round, players, letters) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[convergence] skipping plot (matplotlib unavailable: {e})")
        return

    rounds = sorted(phi_by_round)
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in players:
        ax.plot(rounds, [phi_by_round[t][p] for t in rounds], marker="o",
                label=f"{letters[p]} (client {p})")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("federated round")
    ax.set_ylabel(r"exact Shapley $\phi_i$ (mAP50)")
    ax.set_title(f"Contribution stabilization -- {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"shapley_by_round_{label}.png", dpi=150)
    plt.close(fig)
    print(f"[convergence] wrote {out / f'shapley_by_round_{label}.png'}")


def parse_args():
    p = argparse.ArgumentParser(description="Disruption-timing / convergence analysis.")
    p.add_argument("--log_dirs", nargs="+", required=True,
                   help="One or more Shapley log dirs (experiments/<exp>/fl/shapley_logs), "
                        "e.g. runs with different local-epoch counts E.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display label per log dir (e.g. E1 E2 E5).")
    p.add_argument("--local_epochs", nargs="+", type=int, default=None,
                   help="Local epochs E per log dir -- enables the compute-fair "
                        "(round x E) comparison panel.")
    p.add_argument("--test_dir", default="data/neu_data/test", help="Shared test set dir")
    p.add_argument("--out_dir", default="experiments/convergence_analysis")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--class_names", nargs="+", default=["Inclusion", "Patches", "Scratches"])
    p.add_argument("--min_delta", type=float, default=0.005,
                   help="Improvement below this doesn't reset the plateau counter.")
    p.add_argument("--patience", type=int, default=2,
                   help="Consecutive non-improving rounds that define a plateau.")
    p.add_argument("--per_round_shapley", action="store_true",
                   help="Also compute exact Shapley at every round (8 evals/round).")
    p.add_argument("--device", default=None, help="'cpu', '0', ... (auto if omitted)")
    p.add_argument("--imgsz", type=int, default=480)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(log_dirs=a.log_dirs, labels=a.labels, test_dir=a.test_dir, out_dir=a.out_dir,
        local_epochs=a.local_epochs, num_classes=a.num_classes, class_names=a.class_names,
        min_delta=a.min_delta, patience=a.patience,
        do_per_round_shapley=a.per_round_shapley, device=a.device, imgsz=a.imgsz)
