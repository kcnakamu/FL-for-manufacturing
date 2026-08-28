"""
Train the six standalone local models that form the teacher bank.

One independent YOLOv8n per client, trained only on that client's allocated
images. No federation, no aggregation, no weight sharing between them.

Every teacher is built through model.load_model(), which transfers the COCO
pretrained backbone/neck and randomly initialises only the nc=6 head. This is
NOT the same as scripts/train_centralized.py: that script's _build_model()
replaces the whole network with a fresh DetectionModel on an nc mismatch,
silently discarding every pretrained tensor. The transfer is verified against
yolov8n.pt before training starts, so a regression there fails loudly.

All six runs share one frozen hyperparameter set (TEACHER_HP) so that
"trained to convergence" means the same thing in every row of the competence
matrix. Augmentation values are pinned explicitly rather than inherited from
Ultralytics defaults, so a library upgrade cannot desynchronise the six runs.

Usage:
    python scripts/train_local_teachers.py --data_dir data/neu6_data \
        --out_dir experiments/teacher_bank
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import MODEL_PATH, load_model, set_seed  # noqa: E402

NUM_CLASSES = 6
NUM_CLIENTS = 6

# Client index (client_<i> on disk) -> the design's C<n> label.
CLIENT_LABELS = {
    0: "C1 generalist",
    1: "C2 generalist",
    2: "C3 partial specialist",
    3: "C4 partial specialist",
    4: "C5 exclusive owner",
    5: "C6 redundancy control",
}

# Expected allocation, asserted before training so a silently regenerated or
# half-copied dataset cannot produce an incomparable teacher bank.
EXPECTED_TRAIN_IMAGES = {0: 335, 1: 344, 2: 205, 3: 140, 4: 225, 5: 100}
EXPECTED_PER_CLASS = {
    "crazing": 225, "inclusion": 225, "patches": 225,
    "pitted_surface": 225, "rolled-in_scale": 225,
    "scratches": 224,  # one image dropped: carried a pitted_surface box
}

# Frozen hyperparameters -- identical for all six teachers.
TEACHER_HP = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "seed": 0,
    "deterministic": True,
    # Augmentation pinned explicitly (Ultralytics defaults as of 8.4.x).
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5,
    "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5,
    "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
}


def preflight(data_dir: Path) -> None:
    """Fail loudly if the partition on disk isn't the one the teachers assume."""
    problems: list[str] = []
    grand: Counter = Counter()

    for i in range(NUM_CLIENTS):
        cdir = data_dir / f"client_{i}"
        ydir = cdir / "data.yaml"
        if not ydir.exists():
            problems.append(f"client_{i}: no data.yaml at {ydir}")
            continue

        cfg = yaml.safe_load(ydir.read_text())
        if cfg.get("nc") != NUM_CLASSES:
            problems.append(f"client_{i}: data.yaml nc={cfg.get('nc')}, expected {NUM_CLASSES}")

        imgs = sorted((cdir / "images" / "train").glob("*.jpg"))
        if len(imgs) != EXPECTED_TRAIN_IMAGES[i]:
            problems.append(
                f"client_{i}: {len(imgs)} train images, expected {EXPECTED_TRAIN_IMAGES[i]}"
            )
        grand.update(p.stem.rsplit("_", 1)[0] for p in imgs)

    for cls, want in EXPECTED_PER_CLASS.items():
        if grand.get(cls, 0) != want:
            problems.append(f"class '{cls}': {grand.get(cls, 0)} training images, expected {want}")

    val_yaml = data_dir / "val" / "data.yaml"
    if not val_yaml.exists():
        problems.append(f"missing centralized val set at {val_yaml}")
    else:
        n_val = len(list((data_dir / "val" / "images").glob("*.jpg")))
        if n_val != 180:
            problems.append(f"centralized val has {n_val} images, expected 180")

    if problems:
        raise ValueError(
            "Partition does not match what the teacher bank expects:\n  - "
            + "\n  - ".join(problems)
        )
    print(f"[OK] Preflight: 6 clients, per-class totals {dict(sorted(grand.items()))}")


def verify_coco_transfer(model: YOLO) -> None:
    """Confirm the COCO backbone actually survived the nc=6 head rebuild.

    load_model() copies every shape-matching pretrained tensor, but a regression
    there (or accidentally routing through train_centralized._build_model) would
    yield a randomly initialised network that still trains and still produces a
    plausible-looking competence matrix. Compare against yolov8n.pt directly.
    """
    ref = YOLO(MODEL_PATH)
    got = model.model.model[0].conv.weight.detach().cpu()
    want = ref.model.model[0].conv.weight.detach().cpu()
    if got.shape != want.shape or not torch.allclose(got, want):
        raise ValueError(
            "COCO pretrained weights were NOT transferred: the first backbone conv "
            "differs from yolov8n.pt. Refusing to train a randomly initialised teacher."
        )
    if model.model.nc != NUM_CLASSES:
        raise ValueError(f"Model head has nc={model.model.nc}, expected {NUM_CLASSES}")
    print(f"[OK] COCO backbone verified against {MODEL_PATH}; head is nc={NUM_CLASSES}")


def _freeze(path: Path) -> None:
    """Make a checkpoint read-only -- these teachers must never be overwritten."""
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def train_one(cid: int, data_dir: Path, out_dir: Path, device: str, workers: int,
              epochs: int, imgsz: int) -> dict:
    label = CLIENT_LABELS[cid]
    print(f"\n{'=' * 70}\n[client_{cid}] {label} -- training\n{'=' * 70}")

    hp = dict(TEACHER_HP)
    hp["epochs"] = epochs
    hp["imgsz"] = imgsz

    # Seed before building so the random head init is reproducible across clients.
    set_seed(hp["seed"])
    model = load_model(num_classes=NUM_CLASSES)
    verify_coco_transfer(model)

    run_dir = out_dir / "runs"
    model.train(
        data=str((data_dir / f"client_{cid}" / "data.yaml").resolve()),
        project=str(run_dir.resolve()),
        name=f"client_{cid}",
        exist_ok=True,
        workers=workers,
        device=device,
        verbose=False,
        val=True,          # curves only -- see note on last.pt below
        plots=True,
        # Fixed epoch budget: patience must never be able to trigger.
        patience=epochs + 1,
        **hp,
    )

    save_dir = Path(model.trainer.save_dir)

    # Freeze last.pt, not best.pt. Each client's val set IS the centralized val
    # set that the competence matrix scores, so selecting best.pt would tune the
    # teacher on the very images used to measure it.
    src = save_dir / "weights" / "last.pt"
    if not src.exists():
        raise FileNotFoundError(f"client_{cid}: expected {src} after training")

    bank = out_dir / "teacher_bank"
    bank.mkdir(parents=True, exist_ok=True)
    dst = bank / f"local_c{cid + 1}.pt"
    if dst.exists():
        dst.chmod(stat.S_IRUSR | stat.S_IWUSR)  # allow replace on a rerun
    shutil.copy2(src, dst)
    _freeze(dst)

    nc_saved = YOLO(str(dst)).model.nc
    if nc_saved != NUM_CLASSES:
        raise ValueError(f"{dst.name} saved with nc={nc_saved}, expected {NUM_CLASSES}")

    n_train = len(list((data_dir / f"client_{cid}" / "images" / "train").glob("*.jpg")))
    meta = {
        "client": f"client_{cid}",
        "label": label,
        "teacher": dst.name,
        "checkpoint_source": "last.pt",
        "train_images": n_train,
        "num_classes": NUM_CLASSES,
        "coco_pretrained": True,
        "hyperparameters": hp,
        "results_csv": str((save_dir / "results.csv").resolve()),
    }
    (save_dir / "teacher_config.json").write_text(json.dumps(meta, indent=2))
    print(f"[client_{cid}] frozen teacher -> {dst}  ({n_train} train images, nc={nc_saved})")
    return meta


def collect_curves(out_dir: Path) -> None:
    """Merge per-client results.csv into one long-format CSV for overfitting plots."""
    import csv

    rows = []
    for cid in range(NUM_CLIENTS):
        csv_path = out_dir / "runs" / f"client_{cid}" / "results.csv"
        if not csv_path.exists():
            print(f"[WARN] no results.csv for client_{cid}")
            continue
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                r = {k.strip(): v for k, v in r.items()}
                rows.append({
                    "client": f"client_{cid}",
                    "label": CLIENT_LABELS[cid],
                    "epoch": r.get("epoch"),
                    "train_box_loss": r.get("train/box_loss"),
                    "train_cls_loss": r.get("train/cls_loss"),
                    "val_box_loss": r.get("val/box_loss"),
                    "val_cls_loss": r.get("val/cls_loss"),
                    "val_mAP50": r.get("metrics/mAP50(B)"),
                    "val_mAP50_95": r.get("metrics/mAP50-95(B)"),
                })
    if not rows:
        return
    out = out_dir / "training_curves.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[INFO] Combined training curves -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the six local teacher models.")
    ap.add_argument("--data_dir", default="data/neu6_data")
    ap.add_argument("--out_dir", default="experiments/teacher_bank")
    ap.add_argument("--device", default="", help="'0', 'cpu', 'mps'. Empty = auto.")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=TEACHER_HP["epochs"])
    ap.add_argument("--imgsz", type=int, default=TEACHER_HP["imgsz"])
    ap.add_argument("--clients", default="all",
                    help="Comma-separated client indices, or 'all'.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preflight(data_dir)

    cids = (list(range(NUM_CLIENTS)) if args.clients == "all"
            else [int(c) for c in args.clients.split(",")])

    metas = [train_one(cid, data_dir, out_dir, args.device, args.workers,
                       args.epochs, args.imgsz) for cid in cids]

    (out_dir / "teacher_bank" / "manifest.json").write_text(json.dumps({
        "num_classes": NUM_CLASSES,
        "checkpoint_source": "last.pt",
        "shared_hyperparameters": {**TEACHER_HP, "epochs": args.epochs, "imgsz": args.imgsz},
        "teachers": metas,
    }, indent=2))

    collect_curves(out_dir)
    print(f"\n[DONE] Teacher bank -> {out_dir / 'teacher_bank'}")
    for m in metas:
        print(f"  {m['teacher']:<14s} {m['label']:<24s} {m['train_images']:>4d} images")


if __name__ == "__main__":
    main()
