"""Knowledge contribution matrix: per-class Shapley values (factories x defect classes).

Manuscript contribution 1: estimate which factories contribute most strongly to
each defect class. The per-class utilities come free from a persistence run --
persistence.py already evaluates all 8 coalitions at every fine-tuning
checkpoint tau and records per-class AP@50 alongside overall mAP50, dumping the
raw records to <out_dir>/records.json. This module is pure post-processing:

  contribution matrix   phi[tau][class or "overall"][player]
  retention matrix      rho[tau][class][player] = phi(tau) / phi(ref_tau)
  KD class weights      w_c ~ contribution of the OFFLINE clients to class c:
                        dropped during adaptation ("lost"), present at t*
                        ("static"), or surviving adaptation ("persistent" --
                        the recommended "irreplaceable knowledge" signal)
                        -> class_retention_weights.json, consumed by
                        adaptation/distill_finetune.py.

Primary CLI (after a persistence run):

    python -m shapley.contribution \
        --records experiments/<exp>/shapley/records.json \
        --out_dir experiments/<exp>/contribution \
        --offline 0 1 --weight_mode lost

Standalone tau=0-only mode (no fine-tuning needed -- reconstructs the 8
coalitions at t* and evaluates each once):

    python -m shapley.contribution --tau0_only \
        --log_dir experiments/<exp>/fl/shapley_logs --t_star 4 \
        --test_dir data/neu_data/test --out_dir experiments/<exp>/contribution

All matrix math is pure (no YOLO / torch) and unit-tested in
shapley/tests/test_contribution.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from shapley.persistence import (
    _letters,
    coalition_label,
    retention_curve,
    shapley_over_checkpoints,
)

# matrix_by_tau[tau][column][player] = phi, column in class_names + ["overall"]
Matrix = Dict[int, Dict[str, Dict[str, float]]]

OVERALL = "overall"


# ======================================================================== #
# Pure matrix math (no YOLO / torch) -- unit tested.
# ======================================================================== #
def per_class_records(records: List[dict], cls_name: str) -> List[dict]:
    """Rewrite each record's utility to that class's AP@50.

    Records that don't report the class (e.g. zero predictions and zero labels
    for it in that eval) are dropped; shapley_over_checkpoints then skips any
    (tau) whose 8-coalition table became incomplete instead of miscomputing it.
    """
    out = []
    for r in records:
        per_class = r.get("per_class") or {}
        if cls_name not in per_class:
            continue
        out.append({"tau": r["tau"], "coalition": r["coalition"],
                    "mAP50": per_class[cls_name]})
    return out


def contribution_matrix(records: List[dict], players: List[str],
                        class_names: List[str]) -> Matrix:
    """Per-tau Shapley values for overall mAP50 and each class's AP@50."""
    players = list(players)
    matrix: Matrix = {}

    phi_overall, _ = shapley_over_checkpoints(records, players)
    for tau, phi in phi_overall.items():
        matrix.setdefault(tau, {})[OVERALL] = dict(phi)

    for cls in class_names:
        phi_cls, skipped = shapley_over_checkpoints(per_class_records(records, cls), players)
        if skipped:
            print(f"[contribution] WARNING: class '{cls}' skipped incomplete "
                  f"checkpoints (tau, n_missing): {skipped}")
        for tau, phi in phi_cls.items():
            matrix.setdefault(tau, {})[cls] = dict(phi)
    return matrix


def retention_matrix(matrix: Matrix, ref_tau: int = 0, eps: float = 1e-6) -> Matrix:
    """Per-column retention rho_i(tau) = phi_i(tau) / phi_i(ref_tau).

    Columns missing at ref_tau are dropped (no reference to normalize by).
    """
    columns = set(matrix.get(ref_tau, {}))
    out: Matrix = {}
    for col in sorted(columns):
        phi_by_tau = {tau: cols[col] for tau, cols in matrix.items() if col in cols}
        rho = retention_curve(phi_by_tau, ref_tau=ref_tau, eps=eps)
        for tau, r in rho.items():
            out.setdefault(tau, {})[col] = r
    return out


def class_retention_weights(matrix: Matrix, offline_players: List[str],
                            class_names: List[str], mode: str = "lost",
                            tau_end: Optional[int] = None) -> Dict[str, float]:
    """KD loss weights per class from the contribution matrix.

    mode="static":     w~_c = sum_{i in offline} max(0, phi_{i,c}(tau=0))
                       -- how much the offline clients contribute to class c at
                       t*, i.e. the model's dependence on them at the moment of
                       disruption.
    mode="lost":       w~_c = sum_{i in offline} max(0, phi_{i,c}(0) - phi_{i,c}(tau_end))
                       -- how much of that contribution DROPPED during C's
                       adaptation. Caveat: a drop can mean genuine forgetting OR
                       that C absorbed the class itself (its own specialty), so
                       this over-weights classes C can already do -- see
                       docs/persistence_results_explained.md.
    mode="persistent": w~_c = sum_{i in offline} max(0, phi_{i,c}(tau_end))
                       -- how much the offline clients STILL contribute after C
                       has fine-tuned, i.e. the knowledge C could NOT absorb on
                       its own -- the sharpest "irreplaceable" signal, and the
                       one most aligned with what KD should protect.

    tau_end defaults to the last checkpoint (used by 'lost' and 'persistent').
    Weights are normalized to sum to 1; if every raw weight is 0 (nothing to
    retain / nothing lost), a uniform distribution is returned.
    """
    if mode not in ("static", "lost", "persistent"):
        raise ValueError(f"Unknown weight mode '{mode}' "
                         "(use 'static', 'lost', or 'persistent').")
    taus = sorted(matrix)
    if not taus or 0 not in matrix:
        raise ValueError("Contribution matrix has no tau=0 entry.")
    if tau_end is None:
        tau_end = taus[-1]
    if mode in ("lost", "persistent") and tau_end not in matrix:
        raise ValueError(f"tau_end={tau_end} not in matrix (have {taus}).")

    raw: Dict[str, float] = {}
    for cls in class_names:
        total = 0.0
        for p in offline_players:
            phi0 = matrix[0].get(cls, {}).get(p)
            if phi0 is None:
                continue
            if mode == "static":
                total += max(0.0, phi0)
            elif mode == "persistent":
                total += max(0.0, matrix[tau_end].get(cls, {}).get(p, 0.0))
            else:  # lost
                phi_end = matrix[tau_end].get(cls, {}).get(p, 0.0)
                total += max(0.0, phi0 - phi_end)
        raw[cls] = total

    s = sum(raw.values())
    if s <= 0.0:
        return {cls: 1.0 / len(class_names) for cls in class_names}
    return {cls: v / s for cls, v in raw.items()}


# ======================================================================== #
# IO / orchestration.
# ======================================================================== #
def run_from_records(records_path: str, out_dir: str, offline: List[str],
                     class_names: List[str], weight_mode: str = "lost",
                     tau_end: Optional[int] = None, ref_tau: int = 0) -> dict:
    with open(records_path) as fh:
        records = json.load(fh)
    players = sorted({p for r in records for p in r["coalition"]})

    matrix = contribution_matrix(records, players, class_names)
    rho = retention_matrix(matrix, ref_tau=ref_tau)
    weights = class_retention_weights(matrix, offline, class_names,
                                      mode=weight_mode, tau_end=tau_end)

    out = Path(out_dir).resolve()  # absolute: avoid Ultralytics runs/detect/ prefix
    out.mkdir(parents=True, exist_ok=True)
    resolved_tau_end = tau_end if tau_end is not None else sorted(matrix)[-1]
    results = {
        "players": players, "letters": _letters(players),
        "offline": offline, "class_names": class_names,
        "weight_mode": weight_mode, "tau_end": resolved_tau_end,
        "source_records": str(records_path),
        "matrix_by_tau": {str(t): v for t, v in matrix.items()},
        "retention_by_tau": {str(t): v for t, v in rho.items()},
        "class_retention_weights": weights,
    }
    _write_outputs(out, results, matrix, rho, weights, players, class_names)
    print(f"[contribution] KD class weights ({weight_mode}): "
          + ", ".join(f"{c}={w:.3f}" for c, w in weights.items()))
    print(f"[contribution] done -> {out}")
    return results


def run_tau0_only(log_dir: str, t_star: int, out_dir: str, test_dir: str,
                  offline: List[str], class_names: List[str],
                  num_classes: int = 3, device: Optional[str] = None,
                  imgsz: int = 480, rule: Optional[str] = None) -> dict:
    """Static contribution matrix at t* -- no fine-tuning, 8 coalition evals."""
    from shapley.evaluate import build_test_yaml, evaluate_checkpoint
    from shapley.logger import load_global, load_manifest, load_round
    from shapley.persistence import _save_pt
    from shapley.reconstruct import reconstruct
    from shapley.shapley import coalitions

    out = Path(out_dir).resolve()  # absolute: avoid Ultralytics runs/detect/ prefix
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(log_dir)
    rule = rule or manifest.get("rule", "fedavg")
    updates, counts = load_round(log_dir, t_star)
    baseline = load_global(log_dir, t_star - 1)
    players = sorted(updates.keys())

    test_yaml = build_test_yaml(test_dir, class_names)
    records: List[dict] = []
    try:
        for S in coalitions(players):
            label = coalition_label(S, players)
            pt = _save_pt(reconstruct(updates, counts, S, baseline=baseline, rule=rule),
                          num_classes, out / "coalitions" / f"{label}.pt")
            m = evaluate_checkpoint(pt, test_yaml, device=device, imgsz=imgsz,
                                    out_dir=str(out / "eval"), name=f"{label}_tau0")
            records.append({"tau": 0, "coalition": list(S), "label": label,
                            "mAP50": m["mAP50"], "per_class": m["per_class_ap50"]})
            print(f"[contribution] {label:>4} tau=0  mAP50={m['mAP50']:.4f}")
    finally:
        Path(test_yaml).unlink(missing_ok=True)

    with open(out / "records.json", "w") as fh:
        json.dump(records, fh, indent=2)
    return run_from_records(str(out / "records.json"), out_dir, offline,
                            class_names, weight_mode="static")


def _write_outputs(out: Path, results: dict, matrix: Matrix, rho: Matrix,
                   weights: Dict[str, float], players: List[str],
                   class_names: List[str]) -> None:
    letters = _letters(players)
    columns = [OVERALL] + list(class_names)

    with open(out / "contribution_matrix.json", "w") as fh:
        json.dump(results, fh, indent=2)

    with open(out / "contribution_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau", "class"] + [f"phi_{letters[p]}({p})" for p in players])
        for tau in sorted(matrix):
            for col in columns:
                if col in matrix[tau]:
                    w.writerow([tau, col] + [matrix[tau][col].get(p, "") for p in players])

    with open(out / "retention_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau", "class"] + [f"rho_{letters[p]}({p})" for p in players])
        for tau in sorted(rho):
            for col in columns:
                if col in rho[tau]:
                    w.writerow([tau, col] + [rho[tau][col].get(p, "") for p in players])

    with open(out / "class_retention_weights.json", "w") as fh:
        json.dump({"weights": weights, "mode": results["weight_mode"],
                   "offline": results["offline"], "tau_end": results["tau_end"],
                   "source_records": results["source_records"]}, fh, indent=2)

    _plot_heatmap(out, matrix, players, letters, class_names, results["tau_end"])


def _plot_heatmap(out: Path, matrix: Matrix, players, letters, class_names,
                  tau_end: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[contribution] skipping heatmap (matplotlib unavailable: {e})")
        return

    columns = [OVERALL] + list(class_names)

    def _grid(tau):
        g = np.full((len(players), len(columns)), np.nan)
        for j, col in enumerate(columns):
            for i, p in enumerate(sorted(players)):
                if col in matrix.get(tau, {}) and p in matrix[tau][col]:
                    g[i, j] = matrix[tau][col][p]
        return g

    g0 = _grid(0)
    have_delta = tau_end in matrix and tau_end != 0
    fig, axes = plt.subplots(1, 2 if have_delta else 1,
                             figsize=(11 if have_delta else 6, 4), squeeze=False)

    panels = [(g0, r"$\phi_{i,c}$ at $\tau=0$ (t*)")]
    if have_delta:
        panels.append((g0 - _grid(tau_end),
                       rf"lost contribution $\phi(0)-\phi({tau_end})$"))

    for ax, (g, title) in zip(axes[0], panels):
        im = ax.imshow(g, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(columns)), columns, rotation=20)
        ax.set_yticks(range(len(players)),
                      [f"{letters[p]} (client {p})" for p in sorted(players)])
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:.3f}", ha="center", va="center",
                            color="white", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(out / "contribution_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[contribution] wrote {out / 'contribution_heatmap.png'}")


def parse_args():
    p = argparse.ArgumentParser(description="Per-class knowledge contribution matrix.")
    p.add_argument("--records", default=None,
                   help="records.json from a persistence run (primary mode).")
    p.add_argument("--out_dir", default="experiments/contribution")
    p.add_argument("--offline", nargs="+", default=["0", "1"],
                   help="Client ids that go offline at the disruption.")
    p.add_argument("--class_names", nargs="+", default=["Crazing", "Inclusion", "Patches", "Pitted_surface", "Rolled-in_scale", "Scratches"])
    p.add_argument("--weight_mode", choices=["static", "lost", "persistent"], default="lost")
    p.add_argument("--tau_end", type=int, default=None,
                   help="Checkpoint for 'lost'/'persistent' weights (default: last).")
    p.add_argument("--ref_tau", type=int, default=0)
    # tau0-only standalone mode
    p.add_argument("--tau0_only", action="store_true",
                   help="Static matrix straight from FL logs (no fine-tuning).")
    p.add_argument("--log_dir", default=None, help="Shapley log dir (tau0_only mode).")
    p.add_argument("--t_star", type=int, default=None, help="Disruption round (tau0_only mode).")
    p.add_argument("--test_dir", default="data/neu6_data/test")
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--rule", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.tau0_only:
        if not (a.log_dir and a.t_star is not None):
            raise SystemExit("--tau0_only requires --log_dir and --t_star.")
        run_tau0_only(a.log_dir, a.t_star, a.out_dir, a.test_dir, a.offline,
                      a.class_names, num_classes=a.num_classes, device=a.device,
                      imgsz=a.imgsz, rule=a.rule)
    else:
        if not a.records:
            raise SystemExit("Provide --records (or use --tau0_only).")
        run_from_records(a.records, a.out_dir, a.offline, a.class_names,
                         weight_mode=a.weight_mode, tau_end=a.tau_end,
                         ref_tau=a.ref_tau)
