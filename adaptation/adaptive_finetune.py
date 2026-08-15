"""Adaptive staged fine-tuning driver: escalate freeze regime on val plateau.

Fine-tunes C (client 2) from the pre-disruption global model in short segments.
After each segment the controller looks at C's validation mAP50 and decides to
continue, escalate (head_only -> neck_head -> optionally full), or stop.

Each segment is a plain Ultralytics run with the stage's LR and native
freeze= list (mid-run unfreezing would leave the trainer's freeze/BN/LR state
stale). warmup_epochs=1 softens the per-segment optimizer restart.

    python -m adaptation.adaptive_finetune \
        --weights experiments/<exp>/fl/final_model/client_2_final.pt \
        --data data/neu_data/client_2/data.yaml \
        --out_dir experiments/<exp>/adaptive

Writes <out_dir>/trace.json: one entry per segment (mode, lr, epochs, val
mAP50, controller decision) -- the "when did it switch" evidence for the paper.
Validation uses C's own val split from its data.yaml, never the shared test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Repo root on sys.path for `model` (works when run as a file too).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_STAGE_LRS = {"head_only": 1e-3, "neck_head": 1e-4, "full": 1e-5}


def run(
    weights: str,
    data: str,
    out_dir: str,
    stages: Optional[List[str]] = None,
    seg_epochs: int = 5,
    patience: int = 2,
    min_delta: float = 0.005,
    max_epochs: int = 100,
    stage_lrs: Optional[Dict[str, float]] = None,
    imgsz: int = 480,
    batch: int = 16,
    device: str = "",
    seed: int = 0,
) -> dict:
    from ultralytics import YOLO

    from adaptation.controller import StageController
    from model import freeze_indices, set_seed

    stage_lrs = stage_lrs or DEFAULT_STAGE_LRS
    # Absolute so Ultralytics doesn't prepend runs/detect/ to a relative project.
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    controller = StageController(stages=stages, patience=patience, min_delta=min_delta)
    current = str(weights)
    best_val, best_weights = float("-inf"), None
    trace: List[dict] = []
    total_epochs = 0
    segment = 0

    while total_epochs < max_epochs:
        mode = controller.mode
        lr = stage_lrs[mode]
        name = f"seg{segment:02d}_{mode}"
        model = YOLO(current)
        model.train(
            data=data,
            epochs=seg_epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr,
            optimizer="SGD",
            warmup_epochs=1,
            workers=0,
            device=device,
            project=str(out),
            name=name,
            # The trainer re-enables requires_grad for params not listed here,
            # so the freeze must go through this arg.
            freeze=freeze_indices(mode),
            seed=seed,
            exist_ok=True,
            verbose=False,
        )
        val = float(model.trainer.metrics.get("metrics/mAP50(B)", float("nan")))
        save_dir = Path(model.trainer.save_dir)
        last_pt = save_dir / "weights" / "last.pt"
        total_epochs += seg_epochs

        if val > best_val:
            best_val, best_weights = val, str(save_dir / "weights" / "best.pt")

        decision = controller.update(val)
        trace.append({
            "segment": segment, "mode": mode, "lr": lr,
            "epoch_range": [total_epochs - seg_epochs + 1, total_epochs],
            "val_mAP50": val, "action": decision.action,
            "next_mode": decision.mode, "reason": decision.reason,
            "weights": str(last_pt),
        })
        print(f"[adaptive] seg{segment:02d} mode={mode} val_mAP50={val:.4f} "
              f"-> {decision.action} ({decision.reason})")

        segment += 1
        if decision.action == "stop":
            break
        current = str(last_pt)

    switch_epochs = [t["epoch_range"][1] for t in trace if t["action"] == "escalate"]
    result = {
        "config": {"weights": str(weights), "data": data, "stages": controller.stages,
                   "seg_epochs": seg_epochs, "patience": patience,
                   "min_delta": min_delta, "max_epochs": max_epochs,
                   "stage_lrs": stage_lrs, "imgsz": imgsz, "batch": batch,
                   "seed": seed},
        "segments": trace,
        "switch_epochs": switch_epochs,
        "total_epochs": total_epochs,
        "best_val_mAP50": best_val,
        "best_weights": best_weights,
        "stopped_early": bool(trace) and trace[-1]["action"] == "stop",
    }
    with open(out / "trace.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[adaptive] done: best val mAP50 {best_val:.4f} ({best_weights}); "
          f"switches at epochs {switch_epochs} -> {out / 'trace.json'}")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive staged fine-tuning for client C.")
    p.add_argument("--weights", required=True, help="Init weights (pre-disruption global .pt).")
    p.add_argument("--data", default="data/neu_data/client_2/data.yaml")
    p.add_argument("--out_dir", default="experiments/adaptive")
    p.add_argument("--stages", nargs="+", default=["head_only", "neck_head"],
                   choices=["head_only", "neck_head", "full"],
                   help="Escalation order (append 'full' to allow full fine-tuning).")
    p.add_argument("--seg_epochs", type=int, default=5, help="Epochs per segment.")
    p.add_argument("--patience", type=int, default=2,
                   help="Non-improving segments before escalating.")
    p.add_argument("--min_delta", type=float, default=0.005,
                   help="Val mAP50 improvement below this counts as a stall.")
    p.add_argument("--max_epochs", type=int, default=100, help="Total epoch budget.")
    p.add_argument("--lr_head_only", type=float, default=DEFAULT_STAGE_LRS["head_only"])
    p.add_argument("--lr_neck_head", type=float, default=DEFAULT_STAGE_LRS["neck_head"])
    p.add_argument("--lr_full", type=float, default=DEFAULT_STAGE_LRS["full"])
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(weights=a.weights, data=a.data, out_dir=a.out_dir, stages=a.stages,
        seg_epochs=a.seg_epochs, patience=a.patience, min_delta=a.min_delta,
        max_epochs=a.max_epochs,
        stage_lrs={"head_only": a.lr_head_only, "neck_head": a.lr_neck_head,
                   "full": a.lr_full},
        imgsz=a.imgsz, batch=a.batch, device=a.device, seed=a.seed)
