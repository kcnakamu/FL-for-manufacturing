import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path
import pandas as pd
from PIL import Image, ImageEnhance
import yaml

BRIGHTNESS_RANGE = (0.7, 1.3)

DEFECT_COLS = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]
# Hardcoded config based on CoatingVision Dataset structure
SPLIT_CONFIG = {
    "test": 50,
    "val_per_client": 20,
    "train_clients": [245, 123, 41],  # must sum to 409
    "neg_test": 0,
    "neg_val_per_client": 2,
    "neg_train_clients": [23, 11, 4],  # must sum to 38
    "seed": 42,
}

assert sum(SPLIT_CONFIG["train_clients"]) + SPLIT_CONFIG["test"] + \
       SPLIT_CONFIG["val_per_client"] * 3 == 519, "Positive split doesn't sum to 519"
assert sum(SPLIT_CONFIG["neg_train_clients"]) + SPLIT_CONFIG["neg_test"] + \
       SPLIT_CONFIG["neg_val_per_client"] * 3 == 44, "Negative split doesn't sum to 44"


def get_class_frequency(coating_vision_dir: Path):
    folder = Path(coating_vision_dir / "detection" / "labels")

    rows = []

    for txt_file in folder.glob("*.txt"):
        class_freq = {}
        with open(txt_file, "r") as f:
            for line in f:
                defect_class = line.strip()[0]
                class_freq[defect_class] = class_freq.get(defect_class, 0) + 1
        
        rows.append({"file": txt_file.name[:-4], **class_freq})

    df = pd.DataFrame(rows).fillna(0).set_index("file")
    return df

def load_negative_stems(coating_vision_dir: Path, detection_img_dir: Path):
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

def filter_positives(df: pd.DataFrame, scratch_only: bool = False) -> list[str]:
    """
    Return stems of images containing at least one surface scratch (class 0).
    If scratch_only=True, exclude images that also have other defect classes.
    """
    positive_images = df[df['0'] != 0]
    if scratch_only:
        positive_images = positive_images[positive_images['1'] == 0]
    return positive_images.index.tolist()


def split_positives(stems: list[str], config: dict) -> dict:
    """
    Shuffle stems (sort then seed for reproducibility) and slice into:
        - test: flat list
        - val: list of 3 lists (one per client, val_per_client each)
        - train: list of 3 lists (sized by train_clients)
    """
    random.seed(config["seed"])
    shuffled = sorted(stems)
    random.shuffle(shuffled)

    test = shuffled[:config["test"]]
    remaining = shuffled[config["test"]:]

    val = []
    cursor = 0
    for _ in range(3):
        val.append(remaining[cursor : cursor + config["val_per_client"]])
        cursor += config["val_per_client"]

    train = []
    for n in config["train_clients"]:
        train.append(remaining[cursor : cursor + n])
        cursor += n

    return {"test": test, "val": val, "train": train}


def split_negatives(neg_stems: list[str], config: dict) -> dict:
    """
    Same shuffling strategy as split_positives. Returns:
        - val: list of 3 lists (neg_val_per_client each)
        - train: list of 3 lists (sized by neg_train_clients)
    No test key — negatives never go in the test set.
    """
    random.seed(config["seed"])
    shuffled = sorted(neg_stems)
    random.shuffle(shuffled)

    val = []
    cursor = 0
    for _ in range(3):
        val.append(shuffled[cursor : cursor + config["neg_val_per_client"]])
        cursor += config["neg_val_per_client"]

    train = []
    for n in config["neg_train_clients"]:
        train.append(shuffled[cursor : cursor + n])
        cursor += n

    return {"val": val, "train": train}

def build_client_splits(pos_splits: dict, neg_splits: dict | None) -> tuple[list[dict], list[str]]:
    """
    Merge positive and optional negative splits into per-client dicts.
    Returns (clients, test_stems) where clients is a list of 3 dicts
    each with 'train' and 'val' keys, and test_stems is the shared held-out set.
    """
    clients = []
    for i in range(3):
        train = pos_splits["train"][i]
        val = pos_splits["val"][i]

        if neg_splits is not None:
            train = train + neg_splits["train"][i]
            val = val + neg_splits["val"][i]

        clients.append({"train": train, "val": val})

    return clients, pos_splits["test"]


def _augment_brightness(src: Path, dst: Path, rng: random.Random) -> None:
    factor = rng.uniform(*BRIGHTNESS_RANGE)
    ImageEnhance.Brightness(Image.open(src)).enhance(factor).save(dst)


def _write_label_class0_only(src: Path, dst: Path) -> None:
    lines = [l for l in src.read_text().splitlines() if l.startswith("0 ")]
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_yolo_split(
    stems: list[str],
    split_name: str,
    client_dir: Path,
    src_img_dir: Path,
    src_label_dir: Path,
    is_negative: bool = False,
    n_aug_copies: int = 0,
):
    """
    Copy images and labels for stems into client_dir/images/{split_name}
    and client_dir/labels/{split_name}. If is_negative, write an empty
    label file instead of copying one. Warn and skip missing images.
    If n_aug_copies > 0, write that many brightness-jittered copies per image
    alongside the originals (training only — don't pass this for val/test).
    """
    img_out = client_dir / "images" / split_name
    lbl_out = client_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SPLIT_CONFIG["seed"])

    for stem in stems:
        src_img = src_img_dir / f"{stem}.jpg"
        if not src_img.exists():
            print(f"[WARN] missing image: {src_img}")
            continue

        shutil.copy(src_img, img_out / f"{stem}.jpg")

        if is_negative:
            (lbl_out / f"{stem}.txt").write_text("")
            src_lbl = None
        else:
            src_lbl = src_label_dir / f"{stem}.txt"
            if not src_lbl.exists():
                print(f"[WARN] missing label: {src_lbl}")
                continue
            _write_label_class0_only(src_lbl, lbl_out / f"{stem}.txt")

        for k in range(n_aug_copies):
            aug_stem = f"{stem}_aug{k}"
            _augment_brightness(src_img, img_out / f"{aug_stem}.jpg", rng)
            if src_lbl is None:
                (lbl_out / f"{aug_stem}.txt").write_text("")
            else:
                _write_label_class0_only(src_lbl, lbl_out / f"{aug_stem}.txt")


def write_dataset_yaml(client_dir: Path, client_id: int):
    """
    Write dataset.yaml to client_dir with relative train/val paths,
    nc, and class names matching your YOLO class indices.
    """
    yaml_content = {
        "path": str(client_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "Surface_Crack"},
    }
    with open(client_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

def build_dataset(
    coating_vision_dir: Path,
    output_dir: Path,
    include_negatives: bool = False,
    scratch_only: bool = False,
    augment_train: bool = False,
):
    """
    Full pipeline: filter → split → write.
    Calls each function in order and writes test/, client_1/, client_2/, client_3/
    under output_dir. Run twice (with/without negatives) for comparable setups.
    """
    src_img_dir = coating_vision_dir / "detection" / "images"
    src_lbl_dir = coating_vision_dir / "detection" / "labels"

    df = get_class_frequency(coating_vision_dir)
    positives = filter_positives(df, scratch_only=scratch_only)
    assert len(positives) == 519, f"Expected 519 positives, got {len(positives)}"

    pos_splits = split_positives(positives, SPLIT_CONFIG)

    neg_splits = None
    if include_negatives:
        neg_stems = load_negative_stems(coating_vision_dir, src_img_dir)
        assert len(neg_stems) == 44, f"Expected 44 negatives, got {len(neg_stems)}"
        neg_splits = split_negatives(neg_stems, SPLIT_CONFIG)

    _, test_stems = build_client_splits(pos_splits, neg_splits)

    # shared test set
    write_yolo_split(test_stems, "test", output_dir / "test", src_img_dir, src_lbl_dir)

    # per-client splits
    for i in range(3):
        client_dir = output_dir / f"client_{i}"

        pos_train = pos_splits["train"][i]
        pos_val = pos_splits["val"][i]

        n_aug = 1 if augment_train else 0
        write_yolo_split(pos_train, "train", client_dir, src_img_dir, src_lbl_dir, n_aug_copies=n_aug)
        write_yolo_split(pos_val, "val", client_dir, src_img_dir, src_lbl_dir)

        if include_negatives:
            write_yolo_split(neg_splits["train"][i], "train", client_dir, src_img_dir, src_lbl_dir, is_negative=True, n_aug_copies=n_aug)
            write_yolo_split(neg_splits["val"][i], "val", client_dir, src_img_dir, src_lbl_dir, is_negative=True)

        write_dataset_yaml(client_dir, client_id=i)


if __name__ == "__main__":
    build_dataset(
        Path('CoatingVision'),
        Path('dataset/coating_aug'),
        include_negatives=False,
        scratch_only=False,
        augment_train=True,
    )