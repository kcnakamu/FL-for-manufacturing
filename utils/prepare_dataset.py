"""
Converts the CoatingVision dataset into per-client YOLO-compatible directories
for Federated Learning experiments.

Pipeline steps:
  1. Split labeled (positive) detection images into per-client train/val + global test
  2. Add defect-free negative images (derived from classification/labels.csv)
  3. Apply per-client brightness augmentation to simulate non-IID lighting
  4. Write data.yaml for each client
  5. Verify no image appears in more than one split

Output layout:
    <output_dir>/
        client_0/
            images/train/  images/val/
            labels/train/  labels/val/
            data.yaml
        client_1/ ...
        test/
            images/  labels/

Usage:
    # 3 clients, non-IID 60/30/10, positives only
    python utils/prepare_dataset.py

    # include negatives and brightness augmentation
    python utils/prepare_dataset.py --negatives --augment

    # 2 clients, equal split, custom output dir
    python utils/prepare_dataset.py --num_clients 2 --split 0.5 0.5 --output_dir data_2client

    # force rebuild + sanity check
    python utils/prepare_dataset.py --negatives --augment --check --force
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

DEFAULT_SOURCE         = "CoatingVision"
DEFAULT_OUTPUT         = "data"
DEFAULT_NUM_CLIENTS    = 3
DEFAULT_SPLIT          = [0.6, 0.3, 0.1]
DEFAULT_TEST_RATIO     = 0.10
DEFAULT_VAL_TOTAL      = 64
DEFAULT_SEED           = 42
DEFAULT_CLASS_NAMES    = ["surface_crack"]
DEFAULT_NEG_TRAIN_FRAC = 0.80
# Per-client brightness (low, high); extended with neutral if more clients than entries
DEFAULT_BRIGHTNESS = [
    (1.15, 1.50),  # client_0 — well-lit
    (0.80, 1.20),  # client_1 — neutral
    (0.50, 0.85),  # client_2 — dim
]
DEFECT_COLS = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]


def copy_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path):
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)


def write_yaml(path: Path, client_path: Path, class_names: list[str]):
    data = {
        "path":  str(client_path.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_negative_stems(coating_vision_dir: Path, detection_img_dir: Path) -> list[str]:
    """Return stems of images that have 0 for all defect columns in classification/labels.csv."""
    csv_path = coating_vision_dir / "classification" / "labels.csv"
    negatives = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if all(int(row[col]) == 0 for col in DEFECT_COLS if col in row):
                    stem = Path(row["file_name"]).stem
                    if (detection_img_dir / f"{stem}.jpg").exists():
                        negatives.append(stem)
            except (ValueError, KeyError):
                continue
    return negatives


def distribute(items: list, weights: list[float]) -> list[list]:
    """Split items into len(weights) groups proportional to weights (remainder to last)."""
    total_weight = sum(weights)
    norm = [w / total_weight for w in weights]
    groups: list[list] = []
    cursor = 0
    for i, w in enumerate(norm):
        n = round(len(items) * w) if i < len(norm) - 1 else len(items) - cursor
        groups.append(items[cursor: cursor + n])
        cursor += n
    return groups


def step_split_positives(
    detection_dir: Path,
    output_dir: Path,
    num_clients: int,
    client_split: list[float],
    test_ratio: float,
    val_total: int,
    seed: int,
    class_names: list[str],
):
    rng = random.Random(seed)
    img_dir = detection_dir / "images"
    lbl_dir = detection_dir / "labels"

    labeled = sorted(
        p.stem for p in lbl_dir.glob("*.txt")
        if (img_dir / (p.stem + ".jpg")).exists()
    )
    print(f"  Found {len(labeled)} labeled (positive) images")
    rng.shuffle(labeled)

    n_test     = round(len(labeled) * test_ratio)
    test_stems = labeled[:n_test]
    remaining  = labeled[n_test:]

    n_val      = min(val_total, len(remaining))
    val_pool   = remaining[:n_val]
    train_pool = remaining[n_val:]
    print(f"  Test: {n_test} | Val pool: {n_val} | Train pool: {len(train_pool)}")

    train_per_client = distribute(train_pool, client_split)

    base, extra = divmod(len(val_pool), num_clients)
    val_per_client = []
    v = 0
    for i in range(num_clients):
        n = base + (1 if i < extra else 0)
        val_per_client.append(val_pool[v: v + n])
        v += n

    for i in range(num_clients):
        print(f"  client_{i}: {len(train_per_client[i])} train | {len(val_per_client[i])} val")
    print(f"  test:     {len(test_stems)}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for i in range(num_clients):
        client_dir = output_dir / f"client_{i}"
        for split_name, stems in [("train", train_per_client[i]), ("val", val_per_client[i])]:
            for stem in stems:
                copy_pair(
                    img_dir / f"{stem}.jpg",
                    lbl_dir / f"{stem}.txt",
                    client_dir / "images" / split_name / f"{stem}.jpg",
                    client_dir / "labels" / split_name / f"{stem}.txt",
                )
        write_yaml(client_dir / "data.yaml", client_dir, class_names)

    test_dir = output_dir / "test"
    for stem in test_stems:
        copy_pair(
            img_dir / f"{stem}.jpg",
            lbl_dir / f"{stem}.txt",
            test_dir / "images" / f"{stem}.jpg",
            test_dir / "labels" / f"{stem}.txt",
        )


def step_add_negatives(
    coating_vision_dir: Path,
    output_dir: Path,
    num_clients: int,
    client_split: list[float],
    neg_train_frac: float,
    seed: int,
):
    detection_img_dir = coating_vision_dir / "detection" / "images"
    stems = load_negative_stems(coating_vision_dir, detection_img_dir)
    print(f"  Found {len(stems)} defect-free images in labels.csv")

    rng = random.Random(seed)
    rng.shuffle(stems)

    for i, client_stems in enumerate(distribute(stems, client_split)):
        n_train     = round(len(client_stems) * neg_train_frac)
        train_stems = client_stems[:n_train]
        val_stems   = client_stems[n_train:]
        client_dir  = output_dir / f"client_{i}"

        for split_name, split_stems in [("train", train_stems), ("val", val_stems)]:
            img_dst = client_dir / "images" / split_name
            lbl_dst = client_dir / "labels" / split_name
            img_dst.mkdir(parents=True, exist_ok=True)
            lbl_dst.mkdir(parents=True, exist_ok=True)

            for stem in split_stems:
                src = detection_img_dir / f"{stem}.jpg"
                if not src.exists():
                    print(f"    WARNING: {src} not found, skipping")
                    continue
                shutil.copy2(src, img_dst / f"{stem}.jpg")
                (lbl_dst / f"{stem}.txt").write_text("")

        print(f"  client_{i}: +{len(train_stems)} train | +{len(val_stems)} val negatives")


def step_augment_brightness(
    output_dir: Path,
    num_clients: int,
    brightness_ranges: list[tuple[float, float]],
    seed: int,
):
    try:
        from PIL import Image, ImageEnhance
    except ImportError:
        print("  WARNING: Pillow not installed — skipping augmentation (pip install Pillow)")
        return

    rng = random.Random(seed)
    total = 0
    for i in range(num_clients):
        client_dir = output_dir / f"client_{i}"
        if not client_dir.exists():
            print(f"  WARNING: {client_dir} not found, skipping")
            continue

        low, high = brightness_ranges[i] if i < len(brightness_ranges) else (0.8, 1.2)
        count = 0
        for split in ("train", "val"):
            img_dir = client_dir / "images" / split
            lbl_dir = client_dir / "labels" / split
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                if img_path.stem.endswith("_aug"):
                    continue
                factor  = rng.uniform(low, high)
                aug_img = img_dir / f"{img_path.stem}_aug.jpg"
                aug_lbl = lbl_dir / f"{img_path.stem}_aug.txt"
                src_lbl = lbl_dir / f"{img_path.stem}.txt"

                img = Image.open(img_path).convert("RGB")
                ImageEnhance.Brightness(img).enhance(factor).save(aug_img, quality=95)
                shutil.copy2(src_lbl, aug_lbl) if src_lbl.exists() else aug_lbl.write_text("")
                count += 1

        print(f"  client_{i} [{low:.2f}–{high:.2f}]: {count} images augmented")
        total += count
    print(f"  Total: {total} augmented images added")


def step_check_splits(output_dir: Path) -> bool:
    stem_to_locs: dict[str, list[str]] = defaultdict(list)
    for client_dir in sorted(output_dir.glob("client_*")):
        for split in ("train", "val"):
            img_dir = client_dir / "images" / split
            if img_dir.exists():
                for img in img_dir.iterdir():
                    stem_to_locs[img.stem].append(f"{client_dir.name}/{split}")
    test_img_dir = output_dir / "test" / "images"
    if test_img_dir.exists():
        for img in test_img_dir.iterdir():
            stem_to_locs[img.stem].append("test")

    duplicates = {k: v for k, v in stem_to_locs.items() if len(v) > 1}
    if duplicates:
        print(f"  FAIL  {len(duplicates)} image(s) appear in multiple splits:")
        for stem, locs in sorted(duplicates.items()):
            print(f"    {stem}: {locs}")
        return False

    counts: dict[str, int] = defaultdict(int)
    for locs in stem_to_locs.values():
        for loc in locs:
            counts[loc] += 1
    total = sum(counts.values())
    print(f"  {'Location':<22}  {'Count':>6}  {'%':>6}")
    print("  " + "-" * 38)
    for client_dir in sorted(output_dir.glob("client_*")):
        for split in ("train", "val"):
            label = f"{client_dir.name}/{split}"
            n = counts.get(label, 0)
            print(f"  {label:<22}  {n:>6}  {n/total*100:>5.1f}%")
    if "test" in counts:
        n = counts["test"]
        print(f"  {'test':<22}  {n:>6}  {n/total*100:>5.1f}%")
    print("  " + "-" * 38)
    print(f"  {'TOTAL':<22}  {total:>6}")
    print("  PASS  No image appears in more than one split.")
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare CoatingVision for Federated Learning with YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",          default=DEFAULT_SOURCE,
                   help="CoatingVision root dir (must contain classification/ and detection/)")
    p.add_argument("--output_dir",      default=DEFAULT_OUTPUT,
                   help="Output root directory")
    p.add_argument("--num_clients",     default=DEFAULT_NUM_CLIENTS, type=int,
                   help="Number of FL clients")
    p.add_argument("--split",           default=DEFAULT_SPLIT, type=float, nargs="+",
                   help="Relative train allocation per client (normalised automatically)")
    p.add_argument("--test_ratio",      default=DEFAULT_TEST_RATIO, type=float,
                   help="Fraction of labeled images held out as global test set")
    p.add_argument("--val_total",       default=DEFAULT_VAL_TOTAL, type=int,
                   help="Total val images distributed across all clients")
    p.add_argument("--seed",            default=DEFAULT_SEED, type=int)
    p.add_argument("--class_names",     default=DEFAULT_CLASS_NAMES, nargs="+",
                   help="YOLO class names written to each data.yaml")

    # Negatives
    p.add_argument("--negatives",       action="store_true",
                   help="Include defect-free images (read from classification/labels.csv)")
    p.add_argument("--neg_train_frac",  default=DEFAULT_NEG_TRAIN_FRAC, type=float,
                   help="Fraction of negatives going to train (rest to val)")

    # Augmentation
    p.add_argument("--augment",         action="store_true",
                   help="Add per-client brightness augmentation to simulate non-IID lighting")
    p.add_argument("--brightness_low",  default=None, type=float, nargs="+",
                   help="Min brightness factor per client (default: 1.15 0.80 0.50 ...)")
    p.add_argument("--brightness_high", default=None, type=float, nargs="+",
                   help="Max brightness factor per client (default: 1.50 1.20 0.85 ...)")

    p.add_argument("--check",           action="store_true",
                   help="Verify no image appears in multiple splits after writing")
    p.add_argument("--force",           action="store_true",
                   help="Delete and re-create output_dir even if it already exists")
    return p.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output_dir)

    if len(args.split) != args.num_clients:
        raise ValueError(f"--split has {len(args.split)} values but --num_clients is {args.num_clients}")

    if output_dir.exists() and not args.force:
        print(f"'{output_dir}' already exists — skipping preparation. Use --force to rebuild.")
        return

    print(f"=== Step 1: Splitting labeled images → {output_dir} ===")
    step_split_positives(
        detection_dir = source_dir / "detection",
        output_dir    = output_dir,
        num_clients   = args.num_clients,
        client_split  = args.split,
        test_ratio    = args.test_ratio,
        val_total     = args.val_total,
        seed          = args.seed,
        class_names   = args.class_names,
    )

    if args.negatives:
        print("\n=== Step 2: Adding negative images ===")
        step_add_negatives(
            coating_vision_dir = source_dir,
            output_dir         = output_dir,
            num_clients        = args.num_clients,
            client_split       = args.split,
            neg_train_frac     = args.neg_train_frac,
            seed               = args.seed,
        )

    if args.augment:
        print("\n=== Step 3: Applying brightness augmentation ===")
        if args.brightness_low and args.brightness_high:
            brightness_ranges = list(zip(args.brightness_low, args.brightness_high))
        else:
            brightness_ranges = list(DEFAULT_BRIGHTNESS)
        while len(brightness_ranges) < args.num_clients:
            brightness_ranges.append((0.80, 1.20))
        step_augment_brightness(output_dir, args.num_clients, brightness_ranges, args.seed)

    if args.check:
        print("\n=== Step 4: Verifying splits ===")
        step_check_splits(output_dir)

    print(f"\nDataset ready: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
