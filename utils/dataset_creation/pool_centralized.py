"""Pool a client partition into one centralized dataset.

The centralized run is the PARITY TARGET the KD result is stated against
("federated + KD reaches centralized performance"), so it must see exactly the
union of what the clients saw -- same images, same labels, same holdout -- and
differ from the federated runs only in who holds the data.

Written as a script because the existing data/neu6_centralized has no
provenance: it was built by hand, so nothing records whether its 1349 training
images really are the union of the six clients' allocations. This one asserts
that, and refuses to write a pool it cannot verify.

Usage:
    python utils/dataset_creation/pool_centralized.py \
        --partition data/neu6s_data --out data/neu6s_centralized
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


def pool(partition: Path, out: Path, num_clients: int = 6,
         exclude: tuple[int, ...] = (), buffer: int = 0, seed: int = 0) -> None:
    """Pool clients into one dataset; `exclude` drops client indices entirely.

    Excluding is how the post-departure condition is built: the departed client's
    DATA is gone from the pool, while its frozen teacher checkpoint survives in
    the bank. Nothing else about the pool changes, so a KD run against this and a
    run against the full pool differ only in the missing client.
    """
    if not partition.is_dir():
        raise SystemExit(f"partition not found: {partition}")
    keep = [c for c in range(num_clients) if c not in exclude]
    if not keep:
        raise SystemExit("every client excluded; nothing to pool")
    if buffer and not exclude:
        raise SystemExit("--buffer only means anything alongside --exclude")

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # train: the union of every client's train split.
    seen: dict[str, str] = {}
    for c in keep:
        cdir = partition / f"client_{c}"
        if not cdir.is_dir():
            raise SystemExit(f"missing {cdir}")
        for img in sorted((cdir / "images" / "train").glob("*.jpg")):
            if img.name in seen:
                raise SystemExit(
                    f"{img.name} appears in both {seen[img.name]} and client_{c}; "
                    "clients must hold disjoint images"
                )
            seen[img.name] = f"client_{c}"
            shutil.copy2(img, out / "images" / "train" / img.name)
            lbl = cdir / "labels" / "train" / f"{img.stem}.txt"
            if not lbl.exists():
                raise SystemExit(f"no label for {img.name} in client_{c}")
            shutil.copy2(lbl, out / "labels" / "train" / lbl.name)

    # A retained memory buffer: `buffer` images kept back from an EXCLUDED client,
    # with their labels. This is the transfer-set variable. Distillation can only
    # move knowledge the teacher demonstrates on images it is shown, so a pool
    # holding none of the departed class may be why KD failed to retain it --
    # sweeping this size separates "distillation cannot do this" from "this
    # transfer set cannot". Sampled deterministically so buffer sizes nest:
    # the 10-image buffer is a subset of the 25-image one.
    for c in exclude:
        if not buffer:
            break
        cdir = partition / f"client_{c}"
        pool_imgs = sorted((cdir / "images" / "train").glob("*.jpg"))
        rng = random.Random(seed)
        order = list(pool_imgs); rng.shuffle(order)
        for img in order[:buffer]:
            if img.name in seen:
                raise SystemExit(f"buffer image {img.name} already pooled")
            seen[img.name] = f"client_{c}(buffer)"
            shutil.copy2(img, out / "images" / "train" / img.name)
            lbl = cdir / "labels" / "train" / f"{img.stem}.txt"
            if not lbl.exists():
                raise SystemExit(f"no label for buffered {img.name}")
            shutil.copy2(lbl, out / "labels" / "train" / lbl.name)

    # val/test: the shared centralized holdout, copied through unchanged.
    for split in ("val", "test"):
        src = partition / split
        for img in sorted((src / "images").glob("*.jpg")):
            shutil.copy2(img, out / "images" / split / img.name)
            lbl = src / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out / "labels" / split / lbl.name)

    names = yaml.safe_load((partition / "client_0" / "data.yaml").read_text())["names"]
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": list(names),
    }, sort_keys=False))

    # Verify by re-reading what was written, not by trusting the copy loop.
    counts = {s: len(list((out / "images" / s).glob("*.jpg"))) for s in ("train", "val", "test")}
    client_total = sum(
        len(list((partition / f"client_{c}" / "images" / "train").glob("*.jpg")))
        for c in keep
    ) + buffer * len(exclude)
    if counts["train"] != client_total:
        raise SystemExit(f"pooled train {counts['train']} != sum of clients {client_total}")
    for split in ("train", "val", "test"):
        n_lbl = len(list((out / "labels" / split).glob("*.txt")))
        if n_lbl != counts[split]:
            raise SystemExit(f"{split}: {counts[split]} images but {n_lbl} labels")
    pooled_train = {p.name for p in (out / "images" / "train").glob("*.jpg")}
    for split in ("val", "test"):
        overlap = pooled_train & {p.name for p in (out / "images" / split).glob("*.jpg")}
        if overlap:
            raise SystemExit(f"{len(overlap)} pooled train images also in {split}")

    print(f"[OK] pooled {counts['train']} train (= sum over clients {keep}), "
          f"{counts['val']} val, {counts['test']} test -> {out}")
    print(f"[OK] labels match images in every split; no train/val or train/test overlap")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", required=True, help="e.g. data/neu6s_data")
    ap.add_argument("--out", required=True, help="e.g. data/neu6s_centralized")
    ap.add_argument("--num_clients", type=int, default=6)
    ap.add_argument("--buffer", type=int, default=0,
                    help="Images kept back from each excluded client, with labels.")
    ap.add_argument("--seed", type=int, default=0, help="Buffer sampling seed.")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="Client indices to drop from the pool, e.g. --exclude 4 "
                         "to build the condition after client_4 departs.")
    a = ap.parse_args()
    pool(Path(a.partition), Path(a.out), a.num_clients, tuple(a.exclude), a.buffer, a.seed)


if __name__ == "__main__":
    main()
