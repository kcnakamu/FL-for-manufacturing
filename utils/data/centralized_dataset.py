"""
Merge the per-client federated dataset folders into one centralized dataset
for benchmarking (train/val pooled across clients + the shared test set).

Edit the paths in __main__ to point at your dataset, then run:
    python utils/data/centralized_dataset.py
"""

from pathlib import Path
import shutil


def write_dataset_yaml(output_dir: Path, class_names: list[str]):
    output_dir = Path(output_dir)
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"data.yaml written to {yaml_path}")
    return yaml_path


def get_centralized_dataset(data_dir: Path, output_dir: Path, class_names: list[str] = None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    client_dirs = sorted(data_dir.glob("client_*"))

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Aggregate per-client train/val splits into the centralized dataset.
    for client_dir in client_dirs:
        for split in ("train", "val"):
            for img in (client_dir / "images" / split).glob("*"):
                shutil.copy2(img, output_dir / "images" / split / img.name)
            for lbl in (client_dir / "labels" / split).glob("*"):
                shutil.copy2(lbl, output_dir / "labels" / split / lbl.name)

    # Copy the shared held-out test set (stored at the data root, not per-client).
    test_dir = data_dir / "test"
    if test_dir.is_dir():
        for img in (test_dir / "images").glob("*"):
            shutil.copy2(img, output_dir / "images" / "test" / img.name)
        for lbl in (test_dir / "labels").glob("*"):
            shutil.copy2(lbl, output_dir / "labels" / "test" / lbl.name)
    else:
        print(f"No test set found at {test_dir}; skipping test split")

    print(f"Centralized dataset written to {output_dir}")

    if class_names is not None:
        write_dataset_yaml(output_dir, class_names)


if __name__ == "__main__":
    get_centralized_dataset(
        data_dir="data/neu_data",
        output_dir="data/neu_centralized",
        class_names=["Inclusion", "Patches", "Scratches"],
    )

