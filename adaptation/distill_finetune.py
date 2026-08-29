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

# neu6 partition order -- must match data/neu6_data/*/data.yaml exactly, since
# lambda_c and the teacher-weight columns are applied by class INDEX.
DEFAULT_CLASS_NAMES = ["Crazing", "Inclusion", "Patches",
                       "Pitted_surface", "Rolled-in_scale", "Scratches"]
NEU3_CLASS_NAMES = ["Inclusion", "Patches", "Scratches"]   # the original 3-class split


def resolve_bank(bank: str, exclude: Optional[List[str]] = None):
    """Expand a teacher-bank directory into (names, checkpoint paths, imgsz).

    `bank` is either a directory holding local_c*.pt or a comma-separated list of
    .pt paths. The bank manifest's imgsz is returned so the caller can check it
    against the student's -- teachers are re-run on the student's augmented
    batches, so training the student at a different imgsz evaluates every teacher
    off its training resolution and quietly degrades the targets.
    """
    b = Path(bank)
    if b.is_dir():
        pts = sorted(b.glob("local_c*.pt"), key=lambda q: int(q.stem.split("_c")[1]))
    else:
        pts = [Path(x.strip()) for x in bank.split(",") if x.strip()]
    if not pts:
        raise FileNotFoundError(f"no teacher checkpoints found in {bank}")

    missing = [q for q in pts if not q.exists()]
    if missing:
        raise FileNotFoundError(f"missing teacher checkpoint(s): {missing}")

    names = [q.stem for q in pts]
    for x in (exclude or []):
        if x not in names:
            raise ValueError(f"--exclude_teachers '{x}' not in the bank {names}")
    keep = [i for i, n in enumerate(names) if n not in (exclude or [])]
    if not keep:
        raise ValueError("every teacher excluded; nothing to distill from")

    bank_imgsz = None
    mf = (b / "manifest.json") if b.is_dir() else None
    if mf and mf.exists():
        bank_imgsz = json.loads(mf.read_text()).get(
            "shared_hyperparameters", {}).get("imgsz")

    # `names` (full, pre-exclusion) is returned too: kd_weights rows are indexed
    # against the complete bank, so `keep` is only meaningful relative to it.
    return ([names[i] for i in keep], [str(pts[i]) for i in keep],
            keep, bank_imgsz, names)


def run(
    weights: str,
    data: str,
    out_dir: str,
    weights_json: Optional[str] = None,
    teacher: Optional[str] = None,
    teacher_bank: Optional[str] = None,
    kd_weights: Optional[str] = None,
    exclude_teachers: Optional[List[str]] = None,
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
    teacher_weights = None
    bank_names: List[str] = []

    if teacher_bank:
        bank_names, teacher, keep, bank_imgsz, all_names = resolve_bank(
            teacher_bank, exclude_teachers)
        if bank_imgsz is not None and int(bank_imgsz) != int(imgsz):
            raise ValueError(
                f"teacher bank was trained at imgsz={bank_imgsz} but the student "
                f"is training at imgsz={imgsz}. Teachers are re-run on the "
                f"student's batches, so this evaluates every teacher off its "
                f"training resolution. Pass --imgsz {bank_imgsz}."
            )
        if kd_weights:
            from adaptation.competence_weights import load_kd_weights, select_teachers
            w, full_w = load_kd_weights(kd_weights, class_names, all_names)
            if exclude_teachers:
                w, teacher_weights, orphaned = select_teachers(w, full_w, keep)
                if orphaned:
                    lost = [class_names[i] for i in orphaned]
                    print(f"[distill] ORPHANED by --exclude_teachers "
                          f"{exclude_teachers}: {lost} -- no surviving teacher "
                          f"knows these, so they are dropped from the KD term.")
            else:
                teacher_weights = full_w
        else:
            w = [1.0 / len(class_names)] * len(class_names)
            print(f"[distill] bank of {len(bank_names)} teachers, no --kd_weights "
                  f"-> UNIFORM ensembling (the competence-blind baseline)")
    else:
        teacher = teacher or weights  # distill from the same global we start from
        if weights_json:
            w = load_class_weights(weights_json, class_names)
        else:
            w = [1.0 / len(class_names)] * len(class_names)
            print("[distill] no --weights_json given -> uniform class weights")

    # Absolute so Ultralytics doesn't prepend runs/detect/ to a relative project.
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    trainer_cls = make_kd_trainer(teacher, w, lam=lam, temperature=temperature,
                                  teacher_conf=teacher_conf,
                                  teacher_weights=teacher_weights)

    config = {
        "weights": str(weights), "teacher": str(teacher), "data": data,
        "teacher_bank": bank_names or None, "kd_weights": kd_weights,
        "excluded_teachers": exclude_teachers or None,
        "teacher_weights": (
            {t: dict(zip(class_names, row)) for t, row in zip(bank_names, teacher_weights)}
            if teacher_weights else None),
        "ensembling": ("competence-weighted" if teacher_weights
                       else "uniform" if bank_names else "single-teacher"),
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
                   help="Single teacher .pt (default: same as --weights).")
    p.add_argument("--teacher_bank", default=None,
                   help="Teacher-bank dir (local_c*.pt) or comma-separated .pt list. "
                        "Overrides --teacher and enables ensemble distillation.")
    p.add_argument("--kd_weights", default=None,
                   help="kd_weights JSON from adaptation.competence_weights. "
                        "Omit with --teacher_bank for uniform (competence-blind) "
                        "ensembling.")
    p.add_argument("--exclude_teachers", nargs="*", default=None,
                   help="Teacher names to withhold entirely, e.g. local_c5. "
                        "Remaining columns are renormalised; classes left with no "
                        "informed teacher are dropped from the KD term.")
    p.add_argument("--data", default="data/neu6_data/client_5/data.yaml")
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
    p.add_argument("--imgsz", type=int, default=640,
                   help="Must match the teacher bank's training imgsz (checked).")
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
        teacher=a.teacher, teacher_bank=a.teacher_bank, kd_weights=a.kd_weights,
        exclude_teachers=a.exclude_teachers, class_names=a.class_names, lam=a.lam,
        temperature=a.temperature, teacher_conf=a.teacher_conf, mode=a.mode,
        epochs=a.epochs, lr=a.lr, imgsz=a.imgsz, batch=a.batch, device=a.device,
        seed=a.seed, save_period=a.save_period, adaptive=a.adaptive,
        seg_epochs=a.seg_epochs, patience=a.patience, min_delta=a.min_delta)
