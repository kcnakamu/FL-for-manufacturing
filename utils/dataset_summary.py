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

        # find all subdirectories inside this folder
        subdirs = [p for p in folder.rglob("*") if p.is_dir()]
        
        for subdir in subdirs:
            count = sum(1 for p in subdir.rglob("*") if p.is_file())
            rel_path = subdir.relative_to(base)
            print(f"{rel_path}: {count} files")
        
        print("")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data")
    args = p.parse_args()

    get_summary(args.source)

if __name__ == "__main__":
    main()