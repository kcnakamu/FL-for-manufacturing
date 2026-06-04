"""
Update the `path` field in each client's data.yaml to match the current
absolute location on disk.

Usage:
    python utils/update_yaml_paths.py data_folder/data
    python utils/update_yaml_paths.py data_folder/data_aug --dry-run
"""

import argparse
from pathlib import Path

import yaml


def update_paths(dataset_dir: Path, dry_run: bool = False) -> int:
    yaml_files = sorted(dataset_dir.glob("client_*/data.yaml"))
    if not yaml_files:
        print(f"No client_*/data.yaml files found under '{dataset_dir}'")
        return 0

    updated = 0
    for yaml_path in yaml_files:
        correct_path = str(yaml_path.parent.resolve())

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        current_path = data.get("path", "")
        if current_path == correct_path:
            print(f"  [ok]      {yaml_path.parent.name}/data.yaml")
            continue

        print(f"  [update]  {yaml_path.parent.name}/data.yaml")
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
    p.add_argument("dataset_dir", help="Dataset root containing client_* subdirectories")
    p.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Directory not found: '{dataset_dir}'")

    update_paths(dataset_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
