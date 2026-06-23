"""
Update the `path` field in each data.yaml to match the current absolute
location on disk.

Works for both dataset layouts:
  - Centralized: a single data.yaml at the dataset root (e.g. data/neu_centralized)
  - Federated:   one data.yaml per client/split subfolder (e.g. data/neu_data with
                 client_*/data.yaml and test/data.yaml)

Every data.yaml found under the given directory has its `path` set to the
absolute path of the directory that contains it.

Usage:
    python utils/data/update_yaml_paths.py data/neu_centralized
    python utils/data/update_yaml_paths.py data/neu_data
    python utils/data/update_yaml_paths.py data/neu_data --dry-run
"""

import argparse
from pathlib import Path

import yaml


def update_paths(dataset_dir: Path, dry_run: bool = False) -> int:
    yaml_files = sorted(dataset_dir.rglob("data.yaml"))
    if not yaml_files:
        print(f"No data.yaml files found under '{dataset_dir}'")
        return 0

    updated = 0
    for yaml_path in yaml_files:
        correct_path = str(yaml_path.parent.resolve())
        rel = yaml_path.relative_to(dataset_dir)

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        current_path = data.get("path", "")
        if current_path == correct_path:
            print(f"  [ok]      {rel}")
            continue

        print(f"  [update]  {rel}")
        print(f"            {current_path!r}")
        print(f"         -> {correct_path!r}")

        if not dry_run:
            data["path"] = correct_path
            with open(yaml_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        updated += 1

    if dry_run and updated:
        print(f"\nDry run — {updated} file(s) would be updated.")
    elif updated:
        print(f"\nUpdated {updated} file(s).")
    else:
        print("\nAll paths already correct.")

    return updated


def main():
    p = argparse.ArgumentParser(
        description="Fix data.yaml paths after moving the dataset to a new location.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "dataset_dir",
        help="Dataset root to search recursively for data.yaml files "
        "(centralized dataset folder or federated root with client_*/ subdirs)",
    )
    p.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Directory not found: '{dataset_dir}'")

    update_paths(dataset_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
