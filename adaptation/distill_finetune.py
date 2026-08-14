"""Fine-tune C with class-retention-weighted knowledge distillation.

Teacher = the pre-disruption global model (frozen); student initializes from
the same weights and fine-tunes on C's local data with the KD term attached:

    total = box + cls + dfl + lambda * sum_c w_c * KD_c

w_c comes from shapley/contribution.py's class_retention_weights.json (classes
whose contribution the offline clients stand to lose get higher retention
pressure). --lam 0 reproduces a plain fine-tune (KD logged but inert) -- use it
as the sanity baseline.

    python -m adaptation.distill_finetune \
        --weights experiments/<exp>/fl/final_model/client_2_final.pt \
        --weights_json experiments/<exp>/contribution/class_retention_weights.json \
        --data data/neu_data/client_2/data.yaml \
        --out_dir experiments/<exp>/distill --mode neck_head --epochs 75

With --adaptive, stage selection is delegated to the StageController
(adaptation/controller.py) and --mode is ignored: training proceeds in short
segments that escalate on validation plateau, each segment carrying the KD loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Repo root on sys.path for `model` (works when run as a file too).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CLASS_NAMES = ["Inclusion", "Patches", "Scratches"]


def run(
    weights: str,
    data: str,
    out_dir: str,
    weights_json: Optional[str] = None,
    teacher: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    lam: float = 1.0,
    temperature: float = 2.0,
    teacher_conf: float = 0.0,
    mode: str = "neck_head",
    epochs: int = 75,
    lr: float = 1e-4,
    imgsz: int = 480,
    batch: int = 16,
    device: str = "",
    seed: int = 0,
    save_period: int = -1,
    adaptive: bool = False,
    seg_epochs: int = 5,
    patience: int = 2,
    min_delta: float = 0.005,
) -> dict:
    from ultralytics import YOLO

    from adaptation.kd import load_class_weights, make_kd_trainer
    from model import freeze_indices, set_seed

    class_names = class_names or DEFAULT_CLASS_NAMES
    teacher = teacher or weights  # default: distill from the same global we start from
    if weights_json:
        w = load_class_weights(weights_json, class_names)
    else:
        w = [1.0 / len(class_names)] * len(class_names)
        print("[distill] no --weights_json given -> uniform class weights")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    trainer_cls = make_kd_trainer(teacher, w, lam=lam, temperature=temperature,
                                  teacher_conf=teacher_conf)

    config = {
        "weights": str(weights), "teacher": str(teacher), "data": data,
        "class_names": class_names, "class_weights": dict(zip(class_names, w)),
        "weights_json": weights_json, "lam": lam, "temperature": temperature,
        "teacher_conf": teacher_conf, "imgsz": imgsz, "batch": batch,
        "seed": seed, "adaptive": adaptive,
    }
    print(f"[distill] class weights: "
          + ", ".join(f"{c}={v:.3f}" for c, v in zip(class_names, w))
          + f"  lam={lam} T={temperature} teacher_conf={teacher_conf}")

    if adaptive:
        from adaptation.controller import StageController
        controller = StageController(patience=patience, min_delta=min_delta)
        current, trace, total = str(weights), [], 0
        best_val, best_weights = float("-inf"), None
        segment = 0
        while total < epochs:
            m = controller.mode
            model = YOLO(current)
            model.train(data=data, epochs=seg_epochs, imgsz=imgsz, batch=batch,
                        lr0=lr, optimizer="SGD", warmup_epochs=1, workers=0,
                        device=device, project=str(out), name=f"seg{segment:02d}_{m}",
                        freeze=freeze_indices(m), seed=seed, exist_ok=True,
                        verbose=False, trainer=trainer_cls)
            val = float(model.trainer.metrics.get("metrics/mAP50(B)", float("nan")))
            save_dir = Path(model.trainer.save_dir)
            total += seg_epochs
            if val > best_val:
                best_val, best_weights = val, str(save_dir / "weights" / "best.pt")
            decision = controller.update(val)
            trace.append({"segment": segment, "mode": m, "val_mAP50": val,
                          "action": decision.action, "reason": decision.reason})
            print(f"[distill] seg{segment:02d} mode={m} val_mAP50={val:.4f} "
                  f"-> {decision.action}")
            segment += 1
            if decision.action == "stop":
                break
            current = str(save_dir / "weights" / "last.pt")
        config.update({"segments": trace, "total_epochs": total,
                       "best_val_mAP50": best_val, "best_weights": best_weights})
    else:
        model = YOLO(str(weights))
        model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr,
                    optimizer="SGD", workers=0, device=device, project=str(out),
                    name=f"kd_{mode}", freeze=freeze_indices(mode), seed=seed,
                    save_period=save_period, exist_ok=True, trainer=trainer_cls)
        save_dir = Path(model.trainer.save_dir)
        config.update({"mode": mode, "epochs": epochs, "lr": lr,
                       "save_dir": str(save_dir),
                       "best_weights": str(save_dir / "weights" / "best.pt")})

    with open(out / "distill_config.json", "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"[distill] done -> {out / 'distill_config.json'}")
    return config


def parse_args():
    p = argparse.ArgumentParser(description="Class-retention-weighted KD fine-tuning for C.")
    p.add_argument("--weights", required=True, help="Student init (pre-disruption global .pt).")
    p.add_argument("--teacher", default=None,
                   help="Teacher .pt (default: same as --weights).")
    p.add_argument("--data", default="data/neu_data/client_2/data.yaml")
    p.add_argument("--weights_json", default=None,
                   help="class_retention_weights.json from shapley/contribution.py "
                        "(uniform weights if omitted).")
    p.add_argument("--class_names", nargs="+", default=DEFAULT_CLASS_NAMES)
    p.add_argument("--out_dir", default="experiments/distill")
    p.add_argument("--lam", type=float, default=1.0, help="KD term weight (0 = plain fine-tune).")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--teacher_conf", type=float, default=0.0,
                   help="If > 0, only anchors the teacher is confident about distill.")
    p.add_argument("--mode", choices=["head_only", "neck_head", "full"], default="neck_head")
    p.add_argument("--epochs", type=int, default=75, help="Epochs (total budget when --adaptive).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_period", type=int, default=-1,
                   help="Checkpoint every K epochs (for persistence analysis of the KD run).")
    p.add_argument("--adaptive", action="store_true",
                   help="Use the StageController instead of a fixed --mode.")
    p.add_argument("--seg_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min_delta", type=float, default=0.005)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(weights=a.weights, data=a.data, out_dir=a.out_dir, weights_json=a.weights_json,
        teacher=a.teacher, class_names=a.class_names, lam=a.lam,
        temperature=a.temperature, teacher_conf=a.teacher_conf, mode=a.mode,
        epochs=a.epochs, lr=a.lr, imgsz=a.imgsz, batch=a.batch, device=a.device,
        seed=a.seed, save_period=a.save_period, adaptive=a.adaptive,
        seg_epochs=a.seg_epochs, patience=a.patience, min_delta=a.min_delta)
