"""
Convert Federated Learning Dataset into centeralized dataset for benchmarking.

Usage:
    python utils/centeralized_dataset.py (Update target dataset in code)
"""

from pathlib import Path
import shutil


def write_dataset_yaml(output_dir: Path, class_names: list[str]):
    output_dir = Path(output_dir)
    yaml_path = output_dir / "dataset.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"dataset.yaml written to {yaml_path}")
    return yaml_path


def get_centeralized_dataset(data_dir: Path, output_dir: Path, class_names: list[str] = None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    client_dirs = sorted(data_dir.glob("client_*"))

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for client_dir in client_dirs:
        for split in ("train", "val"):
            for img in (client_dir / "images" / split).glob("*"):
                shutil.copy2(img, output_dir / "images" / split / img.name)
            for lbl in (client_dir / "labels" / split).glob("*"):
                shutil.copy2(lbl, output_dir / "labels" / split / lbl.name)

    print(f"Centralized dataset written to {output_dir}")

    if class_names is not None:
        write_dataset_yaml(output_dir, class_names)


if __name__ == "__main__":
    get_centeralized_dataset(
        data_dir="dataset/coating_aug",
        output_dir="dataset/coating_aug_centralized",
        class_names=["Surface_Crack"],
    )

