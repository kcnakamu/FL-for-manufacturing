"""Persistence driver: retention curve rho_i(tau) + forgetting proxy (spec 2.4 / 3).

End-to-end flow (assumes an FL run was logged by shapley/logger.py):

  1. Reconstruct all 8 coalition models at the disruption round t*  (retrain-free).
  2. Fine-tune EACH coalition model on C's data, checkpointing every K epochs
     (this C-fine-tuning IS the disruption process being measured).
  3. Evaluate every coalition at every checkpoint: v(S) = mAP50 on the shared test set.
  4. At each checkpoint tau, the 8 utilities -> exact Shapley -> phi_i(tau).
  5. Retention rho_i(tau) = phi_i(tau) / phi_i(t*)   ( = phi_i at tau=0 ).

Outputs (under --out_dir): shapley_by_checkpoint.csv, retention.csv,
forgetting_per_class.csv, results.json, and retention_curve.png (if matplotlib).

The pure assembly math (records -> per-tau Shapley -> retention) is factored out
as `shapley_over_checkpoints` / `retention_curve` and unit-tested without YOLO.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from shapley.logger import load_global, load_manifest, load_round
from shapley.reconstruct import reconstruct
from shapley.shapley import coalitions, exact_shapley, general_shapley

NDArrays = List[np.ndarray]


# ======================================================================== #
# Pure assembly math (no YOLO / torch) -- unit tested.
# ======================================================================== #
def shapley_over_checkpoints(records: List[dict], players) -> Tuple[Dict[int, Dict[str, float]], List[Tuple[int, int]]]:
    """Turn per-(tau, coalition) utilities into Shapley values per checkpoint.

    Args:
        records: list of {"tau": int, "coalition": iterable-of-players, "mAP50": float}.
        players: the player ids (e.g. ["0","1","2"]).

    Returns:
        (phi_by_tau, skipped) where phi_by_tau[tau] = {player: phi}, and skipped is
        a list of (tau, n_missing_coalitions) for checkpoints whose 2^n utility
        table was incomplete (those tau are omitted rather than mis-computed).
    """
    players = list(players)
    full = coalitions(players)
    by_tau: Dict[int, Dict[FrozenSet, float]] = {}
    for r in records:
        by_tau.setdefault(int(r["tau"]), {})[frozenset(r["coalition"])] = float(r["mAP50"])

    phi_by_tau: Dict[int, Dict[str, float]] = {}
    skipped: List[Tuple[int, int]] = []
    for tau in sorted(by_tau):
        util = by_tau[tau]
        missing = [S for S in full if S not in util]
        if missing:
            skipped.append((tau, len(missing)))
            continue
        phi_by_tau[tau] = (exact_shapley(util, players) if len(players) == 3
                           else general_shapley(util, players))
    return phi_by_tau, skipped


def retention_curve(phi_by_tau: Dict[int, Dict[str, float]], ref_tau: int = 0,
                    eps: float = 1e-6) -> Dict[int, Dict[str, float]]:
    """rho_i(tau) = phi_i(tau) / phi_i(ref_tau).

    A near-zero reference contribution (|phi_i(ref)| < eps) makes the ratio
    meaningless, so it is reported as NaN (pick an earlier, still-improving t*
    where contributions are clearly nonzero to avoid this).
    """
    if ref_tau not in phi_by_tau:
        raise ValueError(f"Reference checkpoint tau={ref_tau} not in phi_by_tau "
                         f"(have {sorted(phi_by_tau)}).")
    ref = phi_by_tau[ref_tau]
    out: Dict[int, Dict[str, float]] = {}
    for tau, phi in phi_by_tau.items():
        out[tau] = {p: (phi[p] / ref[p] if abs(ref.get(p, 0.0)) > eps else float("nan"))
                    for p in phi}
    return out


def _letters(players) -> Dict[str, str]:
    """Map player ids to display letters A, B, C, ... in sorted id order."""
    return {p: chr(ord("A") + i) for i, p in enumerate(sorted(players))}


def coalition_label(S, players) -> str:
    """Readable label like 'A', 'AB', 'ABC', or 'none' for the empty coalition."""
    letters = _letters(players)
    lab = "".join(letters[p] for p in sorted(players) if p in S)
    return lab or "none"


# ======================================================================== #
# YOLO-touching orchestration.
# ======================================================================== #
def _save_pt(arrays: NDArrays, num_classes: int, pt_path: Path) -> Path:
    """Load reconstructed weights into a YOLO model and save a .pt checkpoint."""
    from model import load_model, set_parameters
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(num_classes=num_classes)
    set_parameters(model, arrays)
    model.save(str(pt_path))
    return pt_path


def _finetune_on_c(pt_path: Path, c_data: str, mode: str, epochs: int, lr: float,
                   save_period: int, imgsz: int, device: str,
                   project: Path, name: str, seed: int = 0) -> Path:
    """Fine-tune one coalition model on C's data; return the run's save_dir."""
    from ultralytics import YOLO
    from model import apply_freeze
    model = YOLO(str(pt_path))
    apply_freeze(model, mode)
    model.train(
        data=c_data,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        lr0=lr,
        optimizer="SGD",
        workers=0,
        device=device,
        project=str(project),
        name=name,
        freeze=[],
        seed=seed,
        save_period=save_period,
        exist_ok=True,
        verbose=False,
    )
    return Path(model.trainer.save_dir)


def _checkpoints(run_dir: Path, epochs: int) -> List[Tuple[int, Path]]:
    """List (tau, path) fine-tuning checkpoints, tau>0, sorted by tau.

    Uses Ultralytics' save_period epoch checkpoints (weights/epochN.pt) plus
    last.pt as the final tau=epochs point.
    """
    weights = run_dir / "weights"
    out: Dict[int, Path] = {}
    for p in weights.glob("epoch*.pt"):
        m = re.search(r"epoch(\d+)", p.name)
        if m:
            out[int(m.group(1))] = p
    last = weights / "last.pt"
    if last.exists():
        out.setdefault(epochs, last)  # don't clobber an explicit epoch checkpoint
    return sorted(out.items())


def run(
    log_dir: str,
    t_star: int,
    out_dir: str,
    test_dir: str,
    c_data: str,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None,
    mode: str = "neck_head",
    epochs: int = 60,
    lr: float = 1e-4,
    save_period: int = 10,
    imgsz: int = 480,
    device: Optional[str] = None,
    c_cid: str = "2",
    rule: Optional[str] = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    seed: int = 0,
) -> dict:
    from shapley.evaluate import build_test_yaml, evaluate_checkpoint

    class_names = class_names or ["Inclusion", "Patches", "Scratches"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(log_dir)
    rule = rule or manifest.get("rule", "fedavg")

    updates, counts = load_round(log_dir, t_star)
    baseline = load_global(log_dir, t_star - 1)  # global broadcast INTO round t*
    players = sorted(updates.keys())
    letters = _letters(players)
    print(f"[persistence] t*={t_star} rule={rule} players={players} "
          f"(letters {letters}); fine-tuning client C = {c_cid}")

    # The test data.yaml is constant for the whole analysis -- build it once.
    test_yaml = build_test_yaml(test_dir, class_names)

    def _eval(pt, name):
        m = evaluate_checkpoint(pt, test_yaml, device=device, imgsz=imgsz,
                                conf=conf, iou=iou, out_dir=str(out / "eval"), name=name)
        return m

    records: List[dict] = []
    try:
        for S in coalitions(players):
            label = coalition_label(S, players)
            nd = reconstruct(updates, counts, S, baseline=baseline, rule=rule)

            # tau = 0: the reconstructed coalition model (saved once, then evaluated
            # via the same .pt path as every fine-tuning checkpoint).
            pt = _save_pt(nd, num_classes, out / "coalitions" / f"{label}.pt")
            m0 = _eval(pt, f"{label}_tau0")
            records.append({"tau": 0, "coalition": list(S), "label": label,
                            "mAP50": m0["mAP50"], "per_class": m0["per_class_ap50"]})
            print(f"[persistence] {label:>4} tau=0  mAP50={m0['mAP50']:.4f}")

            # Fine-tune this coalition on C, then evaluate each checkpoint.
            run_dir = _finetune_on_c(pt, c_data, mode, epochs, lr, save_period, imgsz,
                                     device or "", out / "finetune", label, seed=seed)
            for tau, ckpt in _checkpoints(run_dir, epochs):
                m = _eval(ckpt, f"{label}_tau{tau}")
                records.append({"tau": tau, "coalition": list(S), "label": label,
                                "mAP50": m["mAP50"], "per_class": m["per_class_ap50"]})
                print(f"[persistence] {label:>4} tau={tau}  mAP50={m['mAP50']:.4f}")
    finally:
        Path(test_yaml).unlink(missing_ok=True)

    # ---- assemble Shapley + retention ----
    phi_by_tau, skipped = shapley_over_checkpoints(records, players)
    if skipped:
        print(f"[persistence] WARNING: skipped incomplete checkpoints (tau, n_missing): {skipped}")
    if 0 not in phi_by_tau:
        raise RuntimeError("No complete tau=0 utility table -> cannot form retention ratios.")
    rho = retention_curve(phi_by_tau, ref_tau=0)

    # ---- per-class forgetting proxy from the full-coalition trajectory ----
    full_label = coalition_label(frozenset(players), players)
    forgetting = {r["tau"]: r["per_class"] for r in records if r["label"] == full_label}

    results = {
        "t_star": t_star, "rule": rule, "players": players, "letters": letters,
        "c_cid": c_cid, "class_names": class_names,
        "config": {"mode": mode, "epochs": epochs, "lr": lr,
                   "save_period": save_period, "imgsz": imgsz, "seed": seed},
        "phi_by_tau": {str(t): v for t, v in phi_by_tau.items()},
        "retention": {str(t): v for t, v in rho.items()},
        "forgetting_per_class": {str(t): v for t, v in forgetting.items()},
        "skipped_checkpoints": skipped,
    }
    _write_outputs(out, players, letters, c_cid, phi_by_tau, rho, forgetting, results)
    print(f"[persistence] done -> {out}")
    return results


def _write_outputs(out: Path, players, letters, c_cid, phi_by_tau, rho, forgetting, results) -> None:
    with open(out / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    taus = sorted(phi_by_tau)
    with open(out / "shapley_by_checkpoint.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau"] + [f"phi_{letters[p]}({p})" for p in players])
        for t in taus:
            w.writerow([t] + [phi_by_tau[t][p] for p in players])

    with open(out / "retention.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau"] + [f"rho_{letters[p]}({p})" for p in players])
        for t in taus:
            w.writerow([t] + [rho[t][p] for p in players])

    if forgetting:
        classes = sorted({c for d in forgetting.values() for c in d})
        with open(out / "forgetting_per_class.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tau"] + [f"AP50_{c}" for c in classes])
            for t in sorted(forgetting):
                w.writerow([t] + [forgetting[t].get(c, "") for c in classes])

    _plot_retention(out, players, letters, c_cid, rho)


def _plot_retention(out: Path, players, letters, c_cid, rho) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[persistence] skipping plot (matplotlib unavailable: {e})")
        return

    taus = sorted(rho)
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in sorted(players):
        # The dropped clients (A, B) are the story; C is the one fine-tuning.
        style = "--" if p == c_cid else "-"
        label = f"{letters[p]}" + (" (C, fine-tuning)" if p == c_cid else "")
        ax.plot(taus, [rho[t][p] for t in taus], style, marker="o", label=label)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("C fine-tuning epochs (tau)")
    ax.set_ylabel(r"retention  $\rho_i(\tau)=\phi_i(\tau)/\phi_i(t^*)$")
    ax.set_title("Contribution persistence as C fine-tunes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "retention_curve.png", dpi=150)
    plt.close(fig)
    print(f"[persistence] wrote {out / 'retention_curve.png'}")


def parse_args():
    p = argparse.ArgumentParser(description="Shapley contribution-persistence driver.")
    p.add_argument("--log_dir", required=True, help="Shapley log dir from an FL run "
                   "(experiments/<exp>/fl/shapley_logs)")
    p.add_argument("--t_star", type=int, required=True, help="Disruption round t* "
                   "(A/B go offline). Uses global_round_{t*-1} as v(emptyset) baseline.")
    p.add_argument("--out_dir", default="experiments/shapley", help="Output directory")
    p.add_argument("--test_dir", default="data/neu_data/test", help="Shared test set dir")
    p.add_argument("--c_data", default="data/neu_data/client_2/data.yaml",
                   help="C's data.yaml (the client that keeps fine-tuning)")
    p.add_argument("--num_classes", type=int, default=3,
                   help="Detection classes used in the FL run (must match it).")
    p.add_argument("--class_names", nargs="+", default=["Inclusion", "Patches", "Scratches"])
    p.add_argument("--mode", choices=["head_only", "neck_head", "full"], default="neck_head",
                   help="Freeze regime for the C fine-tuning of every coalition.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_period", type=int, default=10, help="Checkpoint every K epochs (the tau grid).")
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--device", default=None, help="'cpu', '0', ... (auto if omitted)")
    p.add_argument("--c_cid", default="2", help="Client id of C (the fine-tuning client).")
    p.add_argument("--rule", default=None, help="Override aggregation rule (else read from manifest).")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the C fine-tuning of every coalition "
                        "(match the FL run's seed to repeat a full experiment).")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(log_dir=a.log_dir, t_star=a.t_star, out_dir=a.out_dir, test_dir=a.test_dir,
        c_data=a.c_data, num_classes=a.num_classes, class_names=a.class_names,
        mode=a.mode, epochs=a.epochs, lr=a.lr, save_period=a.save_period,
        imgsz=a.imgsz, device=a.device, c_cid=a.c_cid, rule=a.rule,
        conf=a.conf, iou=a.iou, seed=a.seed)
