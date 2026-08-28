"""
Split the NEU Surface Defect Database into federated client folders.

Reads NEU-DET (VOC-format annotations) and writes per-client train/val splits
plus shared holdout sets in Ultralytics YOLO format (images/{train,val} +
labels/{train,val} + data.yaml).

Two partitions are available, selected with --preset:

  neu3  (default)  The original 3-class / 3-client split: Inclusion, Patches,
        Scratches. 15 test images per class; each client carries its own
        10-per-class validation set. Unchanged -- byte-identical output to the
        version of this script that only supported this split.

  neu6  6-class / 6-client split covering every NEU-DET class. Holdout is taken
        FIRST (45 test + 30 val per class, both class-balanced and centralized),
        leaving exactly 225 images per class to distribute to clients. All
        clients validate on the same shared validation set, so no client
        training images are spent on validation.

Usage:
    python utils/dataset_creation/split_neu_data.py --src data/NEU-DET-SOURCE --out data/neu_data
    python utils/dataset_creation/split_neu_data.py --src data/NEU-DET-SOURCE --out data/neu6_data --preset neu6
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

# --------------------------------------------------------------------------
# Split presets
#
# Class keys are the NEU-DET source directory names, lowercased (so
# "pitted_surface" and "rolled-in_scale", not "pitted"/"rolled_in"). Each
# train_per_client list is indexed by client number, matching the client_<i>
# output folders and the cid passed to client.py.
# --------------------------------------------------------------------------

NEU3_PRESET = {
    "classes": {
        "inclusion": 0,
        "patches":   1,
        "scratches": 2,
    },
    "class_names": ["Inclusion", "Patches", "Scratches"],
    "client_labels": ["Factory A", "Factory B", "Factory C (scarce)"],
    "seed": 42,
    "test_per_class": 15,
    "val_mode": "per_client",
    "val_per_client_per_class": 10,
    "train_per_client": {
        "inclusion": [202, 46,  7],
        "patches":   [143, 104, 8],
        "scratches": [112, 78, 65],
    },
}

# 6-class / 6-client partition.
#
# Client numbering: the experiment design calls these C1..C6; on disk they are
# the zero-based client_0..client_5 folders that client.py expects. C1 == client_0,
# ..., C6 == client_5. The exclusive-owner client C5 is therefore index 4.
NEU6_PRESET = {
    "classes": {
        "crazing":         0,
        "inclusion":       1,
        "patches":         2,
        "pitted_surface":  3,
        "rolled-in_scale": 4,
        "scratches":       5,
    },
    "class_names": [
        "Crazing", "Inclusion", "Patches",
        "Pitted_surface", "Rolled-in_scale", "Scratches",
    ],
    "client_labels": [
        "C1 generalist",
        "C2 generalist",
        "C3 partial specialist",
        "C4 partial specialist",
        "C5 exclusive owner",
        "C6 redundancy control",
    ],
    "seed": 42,
    "test_per_class": 45,
    "val_mode": "centralized",
    "val_per_class": 30,
    # Every class must distribute exactly this many images across the clients.
    "train_per_class_total": 225,
    "train_per_client": {
        #                    C1   C2   C3   C4   C5  C6
        "crazing":         [ 50,  40,   0, 110,   0, 25],
        "inclusion":       [ 60, 110,   0,  30,   0, 25],
        "patches":         [ 90, 110,   0,   0,   0, 25],
        "pitted_surface":  [  0,   0,   0,   0, 225,  0],
        "rolled-in_scale": [ 50,   0, 175,   0,   0,  0],
        # 224, not 225: one scratches image is dropped from the pool (see
        # expected_pool_exclusions below). C2 absorbs the shortfall.
        "scratches":       [ 85,  84,  30,   0,   0, 25],
    },
    # Exclusive-owner condition the experiment depends on: this class may only
    # appear in this client index. Mapped by client index, so C5 -> 4.
    "exclusive": {"pitted_surface": 4},
    # NEU-DET files each image under one class folder, but its XML may annotate
    # other classes too. An image outside pitted_surface/ that carries a
    # pitted_surface box would hand pitted supervision to a non-owner client, so
    # such images are dropped from the pool entirely rather than relabelled.
    # Declared here so the resulting shortfall is explicit and asserted: if the
    # source data changes and this count no longer matches, the run fails.
    "expected_pool_exclusions": {"scratches": 1},
}

SPLIT_PRESETS = {
    "neu3": NEU3_PRESET,
    "neu6": NEU6_PRESET,
}

DEFAULT_PRESET = "neu3"

# Backwards compatibility: older docs and scripts refer to SPLIT_CONFIG, which
# was the 3-class config when that was the only one.
SPLIT_CONFIG = NEU3_PRESET

# NEU-DET ships 300 images per class (240 train + 60 validation).
IMAGES_PER_CLASS = 300


def _require(condition: bool, message: str) -> None:
    """Assert `condition`, raising loudly if it fails.

    Deliberately a raise and not a bare `assert`: these checks guard the
    experiment's core invariants (per-class totals, exclusive ownership) and
    must not vanish when Python is run with -O.
    """
    if not condition:
        raise ValueError(f"Invalid split preset: {message}")


def num_clients_of(preset: dict) -> int:
    widths = {len(counts) for counts in preset["train_per_client"].values()}
    _require(
        len(widths) == 1,
        f"train_per_client rows have differing client counts: {sorted(widths)}",
    )
    return widths.pop()


def validate_preset(name: str, preset: dict) -> None:
    """Fail loudly if a preset violates the invariants the experiment relies on."""
    tag = f"'{name}'"
    classes = preset["classes"]
    train = preset["train_per_client"]

    _require(
        sorted(classes.values()) == list(range(len(classes))),
        f"{tag}: class IDs must be contiguous from 0, got {sorted(classes.values())}",
    )
    _require(
        len(preset["class_names"]) == len(classes),
        f"{tag}: {len(preset['class_names'])} class_names for {len(classes)} classes",
    )
    _require(
        set(train) == set(classes),
        f"{tag}: train_per_client keys {sorted(train)} != classes {sorted(classes)}",
    )

    n_clients = num_clients_of(preset)
    _require(
        len(preset["client_labels"]) == n_clients,
        f"{tag}: {len(preset['client_labels'])} client_labels for {n_clients} clients",
    )

    for cls, counts in train.items():
        _require(
            all(isinstance(n, int) and n >= 0 for n in counts),
            f"{tag}: class '{cls}' has a negative or non-integer count: {counts}",
        )

    # Every class must sum to exactly the configured per-class training total,
    # less any images deliberately excluded from that class's pool.
    expected_total = preset.get("train_per_class_total")
    exclusions = preset.get("expected_pool_exclusions", {})
    _require(
        set(exclusions) <= set(classes),
        f"{tag}: expected_pool_exclusions names unknown classes: "
        f"{sorted(set(exclusions) - set(classes))}",
    )
    if expected_total is not None:
        for cls, counts in train.items():
            want = expected_total - exclusions.get(cls, 0)
            _require(
                sum(counts) == want,
                f"{tag}: class '{cls}' sums to {sum(counts)} across clients, "
                f"expected exactly {want} "
                f"({expected_total} less {exclusions.get(cls, 0)} pool exclusion(s)) "
                f"(counts: {counts})",
            )

    # Exclusive ownership: the class must appear in exactly one client.
    for cls, owner in preset.get("exclusive", {}).items():
        _require(cls in train, f"{tag}: exclusive class '{cls}' is not in train_per_client")
        _require(
            0 <= owner < n_clients,
            f"{tag}: exclusive owner index {owner} for '{cls}' is out of range",
        )
        offenders = {i: n for i, n in enumerate(train[cls]) if n and i != owner}
        _require(
            not offenders,
            f"{tag}: '{cls}' must appear ONLY in client_{owner}, but also appears in "
            + ", ".join(f"client_{i} ({n})" for i, n in sorted(offenders.items())),
        )
        _require(
            train[cls][owner] > 0,
            f"{tag}: exclusive owner client_{owner} has no '{cls}' images",
        )

    # Holdout + client allocation must fit inside the source data.
    val_total = (
        preset["val_per_class"]
        if preset["val_mode"] == "centralized"
        else preset["val_per_client_per_class"] * n_clients
    )
    for cls, counts in train.items():
        needed = preset["test_per_class"] + val_total + sum(counts)
        available = IMAGES_PER_CLASS - exclusions.get(cls, 0)
        _require(
            needed <= available,
            f"{tag}: class '{cls}' needs {needed} images but only {available} are "
            f"available ({IMAGES_PER_CLASS} per class less "
            f"{exclusions.get(cls, 0)} pool exclusion(s))",
        )


for _name, _preset in SPLIT_PRESETS.items():
    validate_preset(_name, _preset)


def _normalize_class(name: str) -> str:
    return name.strip().lower()


def parse_voc_xml(
    xml_path: Path, classes: dict, img_w: int = 200, img_h: int = 200
) -> list[str]:
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
        if cls not in classes:
            continue
        class_id = classes[cls]

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


def build_annotation_index(src: Path) -> dict[str, Path]:
    """Map every annotation stem in the source tree to its XML path.

    NEU-DET is not perfectly partitioned: at least one image (crazing_240) lives
    under train/images/ while its annotation sits in validation/annotations/.
    Looking the XML up only within the image's own split silently produces an
    empty label file for such images. Indexing both splits up front avoids that.
    """
    index: dict[str, Path] = {}
    for split_dir in ("train", "validation"):
        ann_dir = src / split_dir / "annotations"
        if not ann_dir.exists():
            continue
        for xml_path in sorted(ann_dir.glob("*.xml")):
            index.setdefault(xml_path.stem, xml_path)
    return index


def collect_stems_by_class(
    src: Path, classes: dict, ann_index: dict[str, Path]
) -> dict[str, list[tuple[Path, Path | None]]]:
    result: dict[str, list[tuple[Path, Path | None]]] = {cls: [] for cls in classes}

    for split_dir in ("train", "validation"):
        img_dir = src / split_dir / "images"
        if not img_dir.exists():
            continue
        for cls_dir in sorted(img_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name.lower()
            if cls not in classes:
                continue  # class not part of this preset
            for img_path in sorted(cls_dir.glob("*.jpg")):
                result[cls].append((img_path, ann_index.get(img_path.stem)))

    return result


def _object_class_names(xml_path: Path | None) -> set[str]:
    if xml_path is None or not xml_path.exists():
        return set()
    root = ET.parse(xml_path).getroot()
    return {_normalize_class(o.findtext("name", "")) for o in root.findall("object")}


def exclude_contaminated(
    stems_by_class: dict, classes: dict, exclusive: dict
) -> tuple[dict, dict]:
    """Drop images carrying boxes of an exclusively-owned class they don't belong to.

    Such an image would hand the exclusive class's supervision to whichever client
    it landed in, breaking the exclusive-owner condition. Dropping it from the pool
    before the shuffle keeps the label files faithful to the source annotations --
    the alternative would be silently rewriting an annotation.

    Returns (filtered_stems_by_class, {class: [(image_name, offending_classes)]}).
    """
    if not exclusive:
        return stems_by_class, {}

    owned = set(exclusive)
    filtered: dict = {}
    dropped: dict = {}

    for cls, stems in stems_by_class.items():
        if cls in owned:
            filtered[cls] = stems  # the owner's own images are the point
            continue
        keep, lost = [], []
        for img_path, xml_path in stems:
            offending = _object_class_names(xml_path) & owned
            if offending:
                lost.append((img_path.name, sorted(offending)))
            else:
                keep.append((img_path, xml_path))
        filtered[cls] = keep
        if lost:
            dropped[cls] = lost

    return filtered, dropped


def split_class(stems: list[tuple[Path, Path | None]], cls: str, config: dict) -> dict:
    test_n   = config["test_per_class"]
    train_ns = config["train_per_client"][cls]
    n_clients = len(train_ns)
    centralized_val = config["val_mode"] == "centralized"

    if centralized_val:
        val_total = config["val_per_class"]
    else:
        val_per_client = config["val_per_client_per_class"]
        val_total = val_per_client * n_clients

    total_needed = test_n + val_total + sum(train_ns)
    if len(stems) < total_needed:
        raise ValueError(
            f"Class '{cls}': need {total_needed} images but only have {len(stems)}"
        )

    # NOTE: the RNG is re-seeded here on every call, i.e. once per class. All
    # classes therefore shuffle from the identical seed state, so their orderings
    # are correlated rather than independent. This is intentional and defines the
    # fixed, reproducible split -- do NOT change it (e.g. seeding once for the whole
    # run) unless you intend to change which images land in each client/test split.
    random.seed(config["seed"])
    shuffled = sorted(stems, key=lambda x: x[0].name)
    random.shuffle(shuffled)

    # Holdout is taken first, so the test/val sets are unaffected by any later
    # change to the client allocation.
    cursor = 0
    test = shuffled[cursor : cursor + test_n]
    cursor += test_n

    if centralized_val:
        val = shuffled[cursor : cursor + val_total]
        cursor += val_total
    else:
        val = []
        for _ in range(n_clients):
            val.append(shuffled[cursor : cursor + val_per_client])
            cursor += val_per_client

    train = []
    for n in train_ns:
        train.append(shuffled[cursor : cursor + n])
        cursor += n

    return {"test": test, "val": val, "train": train}


def _write_pair(
    img_path: Path,
    xml_path: Path | None,
    img_out: Path,
    lbl_out: Path,
    classes: dict,
) -> None:
    """Copy one image and write its converted YOLO label.

    Labels are a faithful conversion of the source XML; images whose annotations
    would violate the exclusive-owner condition are removed from the pool up
    front by exclude_contaminated(), never rewritten here.
    """
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    shutil.copy(img_path, img_out / img_path.name)

    if xml_path is not None and xml_path.exists():
        lines = parse_voc_xml(xml_path, classes)
    else:
        print(f"[WARN] missing annotation for {img_path.name}")
        lines = []

    label_text = "\n".join(lines) + ("\n" if lines else "")
    (lbl_out / f"{img_path.stem}.txt").write_text(label_text)


def write_data_yaml(
    directory: Path, class_names: list[str], is_holdout: bool = False
) -> None:
    if is_holdout:
        data = {
            "path": str(directory.resolve()),
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": len(class_names),
            "names": class_names,
        }
    else:
        data = {
            "path": str(directory.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": class_names,
        }
    with open(directory / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def print_summary(splits_by_class: dict, config: dict, preset_name: str) -> None:
    classes = list(config["classes"].keys())
    n_clients = num_clients_of(config)
    labels = config["client_labels"]
    centralized_val = config["val_mode"] == "centralized"

    name_w = max(len(c) for c in classes) + 2
    row_w = max(len(f"client_{i} ({labels[i]})") for i in range(n_clients)) + 2
    row_w = max(row_w, len("TRAIN TOTAL (per class)") + 2)

    def line(label: str, values: list[int]) -> str:
        cells = "".join(f"{v:>{name_w}d}" for v in values)
        return f"{label:<{row_w}s}{cells}{sum(values):>{name_w}d}"

    header = f"{'':<{row_w}s}" + "".join(f"{c:>{name_w}s}" for c in classes)
    header += f"{'TOTAL':>{name_w}s}"

    print(f"\n=== '{preset_name}' split summary (clients x classes) ===")
    print(header)
    print("-" * len(header))

    for i in range(n_clients):
        counts = [len(splits_by_class[c]["train"][i]) for c in classes]
        print(line(f"client_{i} ({labels[i]})", counts))

    print("-" * len(header))
    totals = [
        sum(len(splits_by_class[c]["train"][i]) for i in range(n_clients))
        for c in classes
    ]
    print(line("TRAIN TOTAL (per class)", totals))

    print()
    if centralized_val:
        print(line("val (shared, central)", [len(splits_by_class[c]["val"]) for c in classes]))
    else:
        for i in range(n_clients):
            print(line(f"val client_{i}", [len(splits_by_class[c]["val"][i]) for c in classes]))
    print(line("test (central)", [len(splits_by_class[c]["test"]) for c in classes]))

    print("-" * len(header))
    grand = []
    for c in classes:
        s = splits_by_class[c]
        n_val = len(s["val"]) if centralized_val else sum(len(v) for v in s["val"])
        grand.append(len(s["test"]) + n_val + sum(len(t) for t in s["train"]))
    print(line("ALLOCATED TOTAL", grand))
    print()


def verify_output(out: Path, config: dict) -> None:
    """Re-read the generated labels and fail loudly if an invariant is violated.

    Checked against the files on disk, not against the config that produced
    them, so a bug in the writing path cannot slip through.
    """
    classes = config["classes"]
    n_clients = num_clients_of(config)
    exclusive = config.get("exclusive", {})
    problems: list[str] = []

    def box_class_ids(label_dir: Path) -> set[int]:
        found = set()
        for txt in label_dir.glob("*.txt"):
            for ln in txt.read_text().split("\n"):
                if ln.strip():
                    found.add(int(ln.split()[0]))
        return found

    train_stems: list[set[str]] = []
    for i in range(n_clients):
        client_dir = out / f"client_{i}"
        stems = {p.stem for p in (client_dir / "images" / "train").glob("*.jpg")}
        train_stems.append(stems)

        # Every image needs a label file.
        for stem in stems:
            if not (client_dir / "labels" / "train" / f"{stem}.txt").exists():
                problems.append(f"client_{i}: {stem}.jpg has no label file")

        # Exclusive classes must not appear in any non-owner client's boxes.
        present = box_class_ids(client_dir / "labels" / "train")
        for cls, owner in exclusive.items():
            if owner != i and classes[cls] in present:
                problems.append(
                    f"client_{i} has '{cls}' boxes but client_{owner} is the exclusive owner"
                )

    # Clients must not share training images.
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            shared = train_stems[i] & train_stems[j]
            if shared:
                problems.append(
                    f"client_{i} and client_{j} share {len(shared)} training images"
                )

    # Holdout must not leak into training.
    all_train = set().union(*train_stems) if train_stems else set()
    for name in ("test", "val"):
        hold_dir = out / name / "images"
        if not hold_dir.is_dir():
            continue
        hold = {p.stem for p in hold_dir.glob("*.jpg")}
        leak = all_train & hold
        if leak:
            problems.append(f"{len(leak)} image(s) appear in both client training and {name}")

    if problems:
        raise ValueError(
            "Generated dataset violates split invariants:\n  - "
            + "\n  - ".join(problems)
        )
    print("[OK] Verified generated dataset: exclusivity, disjointness, label parity.")


def run(
    src: str | Path,
    out: str | Path,
    seed: int | None = None,
    preset: str = DEFAULT_PRESET,
) -> None:
    src = Path(src)
    out = Path(out)

    if preset not in SPLIT_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {', '.join(sorted(SPLIT_PRESETS))}"
        )

    config = dict(SPLIT_PRESETS[preset])
    validate_preset(preset, config)
    if seed is not None:
        config["seed"] = seed

    classes = config["classes"]
    class_names = config["class_names"]
    n_clients = num_clients_of(config)
    centralized_val = config["val_mode"] == "centralized"

    print(f"[INFO] Preset : {preset} ({len(classes)} classes, {n_clients} clients)")
    print(f"[INFO] Source : {src}")
    print(f"[INFO] Output : {out}")
    print(f"[INFO] Seed   : {config['seed']}")
    print(f"[INFO] Val    : {'centralized (shared by all clients)' if centralized_val else 'per-client'}")

    ann_index = build_annotation_index(src)
    stems_by_class = collect_stems_by_class(src, classes, ann_index)
    for cls, stems in stems_by_class.items():
        missing = sum(1 for _, xml in stems if xml is None)
        note = f" ({missing} without annotations)" if missing else ""
        print(f"[INFO] '{cls}': {len(stems)} images found{note}")

    exclusive = config.get("exclusive", {})
    stems_by_class, dropped = exclude_contaminated(stems_by_class, classes, exclusive)
    if exclusive:
        owners = ", ".join(f"'{c}' -> client_{o}" for c, o in sorted(exclusive.items()))
        print(f"[INFO] Exclusive classes: {owners}")
    for cls, lost in sorted(dropped.items()):
        for img_name, offending in lost:
            print(f"[INFO] dropped {img_name} from the '{cls}' pool "
                  f"(carries {'/'.join(offending)} box(es))")

    # The per-client counts are written against a specific number of exclusions;
    # if the source data changes, fail rather than silently short a client.
    expected_excl = config.get("expected_pool_exclusions", {})
    actual_excl = {cls: len(lost) for cls, lost in dropped.items()}
    if actual_excl != dict(expected_excl):
        raise ValueError(
            f"Pool exclusions changed: expected {dict(expected_excl) or '{}'}, "
            f"got {actual_excl or '{}'}. Update 'expected_pool_exclusions' and the "
            f"train_per_client counts in the '{preset}' preset to match."
        )

    splits_by_class = {
        cls: split_class(stems, cls, config)
        for cls, stems in stems_by_class.items()
    }

    # --- shared test set ---
    test_dir = out / "test"
    for cls, splits in splits_by_class.items():
        for img_path, xml_path in splits["test"]:
            _write_pair(img_path, xml_path, test_dir / "images", test_dir / "labels", classes)
    write_data_yaml(test_dir, class_names, is_holdout=True)

    # --- shared validation set (centralized presets only) ---
    if centralized_val:
        val_dir = out / "val"
        for cls, splits in splits_by_class.items():
            for img_path, xml_path in splits["val"]:
                _write_pair(img_path, xml_path, val_dir / "images", val_dir / "labels", classes)
        write_data_yaml(val_dir, class_names, is_holdout=True)

    # --- per-client splits ---
    # Exclusively-owned classes must not reach any other client, not even as a
    # stray box on an image that belongs to a different class folder.
    for i in range(n_clients):
        client_dir = out / f"client_{i}"
        for cls, splits in splits_by_class.items():
            for img_path, xml_path in splits["train"][i]:
                _write_pair(
                    img_path, xml_path,
                    client_dir / "images" / "train",
                    client_dir / "labels" / "train",
                    classes,
                )

        if centralized_val:
            # Option (a): one shared validation set. No client training images are
            # spent on validation -- every client gets an identical copy of the
            # central val set, so client.py's val(split="val") and its
            # images/val image count keep working unchanged.
            for cls, splits in splits_by_class.items():
                for img_path, xml_path in splits["val"]:
                    _write_pair(
                        img_path, xml_path,
                        client_dir / "images" / "val",
                        client_dir / "labels" / "val",
                        classes,
                    )
        else:
            for cls, splits in splits_by_class.items():
                for img_path, xml_path in splits["val"][i]:
                    _write_pair(
                        img_path, xml_path,
                        client_dir / "images" / "val",
                        client_dir / "labels" / "val",
                        classes,
                    )

        write_data_yaml(client_dir, class_names)

    print_summary(splits_by_class, config, preset)
    verify_output(out, config)
    print(f"[DONE] Dataset written to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split NEU Surface Defect dataset into federated learning clients."
    )
    parser.add_argument("--src", required=True, help="Path to NEU-DET directory")
    parser.add_argument("--out", default="neu_data", help="Output directory")
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(SPLIT_PRESETS),
                        help="Which partition to generate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shuffle seed for which images land in each client/test split. "
                             "Vary across runs to test robustness to the specific split. "
                             "Defaults to the preset's seed (42).")
    args = parser.parse_args()
    run(src=args.src, out=args.out, seed=args.seed, preset=args.preset)
