import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

TARGET_CLASSES = {
    "inclusion": 0,
    "patches":   1,
    "scratches": 2,
}

# Maps the on-disk subdirectory name to the canonical class key above.
DIR_TO_CLASS = {
    "inclusion": "inclusion",
    "patches":   "patches",
    "scratches": "scratches",
}

CLASS_NAMES = ["Inclusion", "Patches", "Scratches"]

SPLIT_CONFIG = {
    "seed": 42,
    "test_per_class": 15,
    "val_per_client_per_class": 10,
    "train_per_client": {
        "inclusion": [202, 46,  7],
        "patches":   [143, 104, 8],
        "scratches": [112, 78, 65],
    },
}

NUM_CLIENTS = 3


def _normalize_class(name: str) -> str:
    return name.strip().lower()


def parse_voc_xml(xml_path: Path, img_w: int = 200, img_h: int = 200) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    if size_el is not None:
        try:
            img_w = int(size_el.findtext("width"))
            img_h = int(size_el.findtext("height"))
        except (TypeError, ValueError):
            pass

    lines = []
    for obj in root.findall("object"):
        cls = _normalize_class(obj.findtext("name", ""))
        if cls not in TARGET_CLASSES:
            continue
        class_id = TARGET_CLASSES[cls]

        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))

        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def collect_stems_by_class(src: Path) -> dict[str, list[tuple[Path, Path]]]:
    result: dict[str, list[tuple[Path, Path]]] = {cls: [] for cls in TARGET_CLASSES}

    for split_dir in ("train", "validation"):
        img_dir = src / split_dir / "images"
        ann_dir = src / split_dir / "annotations"
        if not img_dir.exists():
            continue
        for cls_dir in sorted(img_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = DIR_TO_CLASS.get(cls_dir.name)
            if cls is None:
                continue  # skip crazing, pitted_surface, rolled-in_scale
            for img_path in sorted(cls_dir.glob("*.jpg")):
                result[cls].append((img_path, ann_dir / f"{img_path.stem}.xml"))

    return result


def split_class(
    stems: list[tuple[Path, Path]], cls: str, config: dict
) -> dict:
    test_n  = config["test_per_class"]
    val_n   = config["val_per_client_per_class"]
    train_ns = config["train_per_client"][cls]

    total_needed = test_n + val_n * NUM_CLIENTS + sum(train_ns)
    if len(stems) < total_needed:
        raise ValueError(
            f"Class '{cls}': need {total_needed} images but only have {len(stems)}"
        )

    random.seed(config["seed"])
    shuffled = sorted(stems, key=lambda x: x[0].name)
    random.shuffle(shuffled)

    cursor = 0
    test = shuffled[cursor : cursor + test_n]
    cursor += test_n

    val = []
    for _ in range(NUM_CLIENTS):
        val.append(shuffled[cursor : cursor + val_n])
        cursor += val_n

    train = []
    for n in train_ns:
        train.append(shuffled[cursor : cursor + n])
        cursor += n

    return {"test": test, "val": val, "train": train}


def _write_pair(
    img_path: Path,
    xml_path: Path,
    img_out: Path,
    lbl_out: Path,
) -> None:
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    shutil.copy(img_path, img_out / img_path.name)

    if xml_path.exists():
        lines = parse_voc_xml(xml_path)
    else:
        print(f"[WARN] missing annotation: {xml_path}")
        lines = []

    label_text = "\n".join(lines) + ("\n" if lines else "")
    (lbl_out / f"{img_path.stem}.txt").write_text(label_text)


def write_data_yaml(directory: Path, is_test: bool = False) -> None:
    if is_test:
        data = {
            "path": str(directory.resolve()),
            "test": "images",
            "nc": 3,
            "names": CLASS_NAMES,
        }
    else:
        data = {
            "path": str(directory.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 3,
            "names": CLASS_NAMES,
        }
    with open(directory / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def print_summary(splits_by_class: dict) -> None:
    classes = list(TARGET_CLASSES.keys())
    col_w = 18

    header = f"{'':16s}" + "".join(f"{c:>{col_w}s}" for c in classes) + f"{'Total':>{col_w}s}"
    print("\n=== Split Summary ===")
    print(header)
    print("-" * len(header))

    # test
    row = f"{'test':16s}"
    total = 0
    for cls in classes:
        n = len(splits_by_class[cls]["test"])
        row += f"{n:>{col_w}d}"
        total += n
    print(row + f"{total:>{col_w}d}")

    for split_name in ("train", "val"):
        print(f"\n  -- {split_name} --")
        for i in range(NUM_CLIENTS):
            row = f"  client_{i}:{'':7s}"
            total = 0
            for cls in classes:
                n = len(splits_by_class[cls][split_name][i])
                row += f"{n:>{col_w}d}"
                total += n
            print(row + f"{total:>{col_w}d}")

    print()


def run(src: str | Path, out: str | Path) -> None:
    src = Path(src)
    out = Path(out)

    print(f"[INFO] Source : {src}")
    print(f"[INFO] Output : {out}")

    stems_by_class = collect_stems_by_class(src)
    for cls, stems in stems_by_class.items():
        print(f"[INFO] '{cls}': {len(stems)} images found")

    splits_by_class = {
        cls: split_class(stems, cls, SPLIT_CONFIG)
        for cls, stems in stems_by_class.items()
    }

    # --- test set ---
    test_dir = out / "test"
    for cls, splits in splits_by_class.items():
        for img_path, xml_path in splits["test"]:
            _write_pair(img_path, xml_path, test_dir / "images", test_dir / "labels")
    write_data_yaml(test_dir, is_test=True)

    # --- per-client splits ---
    for i in range(NUM_CLIENTS):
        client_dir = out / f"client_{i}"
        for split_name in ("train", "val"):
            for cls, splits in splits_by_class.items():
                for img_path, xml_path in splits[split_name][i]:
                    _write_pair(
                        img_path, xml_path,
                        client_dir / "images" / split_name,
                        client_dir / "labels" / split_name,
                    )
        write_data_yaml(client_dir)

    print_summary(splits_by_class)
    print(f"[DONE] Dataset written to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split NEU Surface Defect dataset into federated learning clients."
    )
    parser.add_argument("--src", required=True, help="Path to NEU-DET directory")
    parser.add_argument("--out", default="neu_data", help="Output directory")
    args = parser.parse_args()
    run(src=args.src, out=args.out)
