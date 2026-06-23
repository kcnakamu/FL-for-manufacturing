# Generate Summary of dataset
from pathlib import Path
import argparse

def get_summary(data_dir):
    base = Path(data_dir)

    folders = [p for p in base.iterdir() if p.is_dir()]
    folders.sort()
    print(f"Found {len(folders)} folders: {', '.join(p.name for p in folders)}\n")

    for folder in folders:
        print(f"--- {folder.name} ---")

        # find leaf directories only (no further subdirs)
        leaf_dirs = [p for p in folder.rglob("*") if p.is_dir() and not any(c.is_dir() for c in p.iterdir())]
        leaf_dirs.sort()

        # group by split name (last path component), pair images vs labels
        splits = {}
        for d in leaf_dirs:
            split = d.name
            category = d.parent.name  # "images" or "labels"
            count = sum(1 for p in d.iterdir() if p.is_file())
            splits.setdefault(split, {})[category] = count

        for split, counts in sorted(splits.items()):
            imgs = counts.get("images", "?")
            lbls = counts.get("labels", "?")
            match = "✓" if imgs == lbls else f"✗ MISMATCH"
            print(f"  {split}: {imgs} images, {lbls} labels  {match}")

        print("")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data")
    args = p.parse_args()

    get_summary(args.source)

if __name__ == "__main__":
    main()