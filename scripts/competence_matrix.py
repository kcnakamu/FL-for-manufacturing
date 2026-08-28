"""
Competence matrix: score every local teacher on the centralized validation set.

Pure inference -- no training, no weight updates. Each of the six teachers is
evaluated on the SAME centralized val set (30 images/class x 6 classes), which
is the only way the rows are comparable.

Per-class AP@50 is read from results.box.ap50 indexed through
results.box.ap_class_index. Never assume the per-class arrays are ordered
0..nc-1: ap_class_index lists only the classes Ultralytics actually scored, so
positional indexing silently misaligns columns whenever a class is absent.
Classes missing from that index scored nothing and are recorded as 0.0.

Usage:
    python scripts/competence_matrix.py --bank experiments/teacher_bank/teacher_bank \
        --val_yaml data/neu6_data/val/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shapley.evaluate import evaluate_checkpoint  # noqa: E402

CLIENT_LABELS = {
    1: "C1 generalist",
    2: "C2 generalist",
    3: "C3 partial specialist",
    4: "C4 partial specialist",
    5: "C5 exclusive owner",
    6: "C6 redundancy control",
}

# The exclusive-owner sanity check: C5 saw only pitted_surface, so it must score
# high there and near-zero everywhere else. If that pattern is absent the column
# indexing is wrong, not the model.
SANITY_TEACHER = 5
SANITY_CLASS = "Pitted_surface"
SANITY_MIN_OWN = 0.10
SANITY_MAX_OTHER = 0.05


def score_bank(bank: Path, val_yaml: Path, device: str, imgsz: int,
               out_dir: Path) -> tuple[list[str], dict]:
    class_names = [str(n) for n in yaml.safe_load(val_yaml.read_text())["names"]]
    matrix: dict[str, dict[str, float]] = {}

    for i in range(1, 7):
        pt = bank / f"local_c{i}.pt"
        if not pt.exists():
            raise FileNotFoundError(f"missing teacher checkpoint: {pt}")
        print(f"[eval] local_c{i}.pt ({CLIENT_LABELS[i]}) on {val_yaml}")
        res = evaluate_checkpoint(
            pt, str(val_yaml), device=device or None, imgsz=imgsz,
            out_dir=str((out_dir / "val_runs").resolve()),
            name=f"local_c{i}", split="val",
        )
        per_class = res["per_class_ap50"]
        # Classes absent from ap_class_index were never scored -> 0.0.
        matrix[f"local_c{i}"] = {c: float(per_class.get(c, 0.0)) for c in class_names}

    return class_names, matrix


def column_stats(class_names: list[str], matrix: dict) -> list[dict]:
    stats = []
    for c in class_names:
        col = sorted(((m, matrix[m][c]) for m in matrix), key=lambda kv: -kv[1])
        top1_model, top1 = col[0]
        top2_model, top2 = col[1]
        stats.append({
            "class": c,
            "argmax_teacher": top1_model,
            "argmax_label": CLIENT_LABELS[int(top1_model[-1])],
            "top1": top1,
            "runner_up": top2_model,
            "top2": top2,
            "margin": top1 - top2,
        })
    return stats


def sanity_check(class_names: list[str], matrix: dict) -> list[str]:
    """Verify the exclusive owner's signature. Misalignment shows up here first."""
    warnings = []
    row = matrix.get(f"local_c{SANITY_TEACHER}")
    if row is None:
        return ["sanity: C5 row missing"]

    own = row.get(SANITY_CLASS, 0.0)
    others = {c: v for c, v in row.items() if c != SANITY_CLASS}
    worst_other, worst_val = max(others.items(), key=lambda kv: kv[1])

    if own < SANITY_MIN_OWN:
        warnings.append(
            f"C5 (pitted-only) scores {own:.4f} on {SANITY_CLASS}, below {SANITY_MIN_OWN}. "
            "Expected high -- suspect misaligned class indexing or a failed teacher."
        )
    if worst_val > SANITY_MAX_OTHER:
        warnings.append(
            f"C5 (pitted-only) scores {worst_val:.4f} on '{worst_other}', a class it "
            f"never saw (limit {SANITY_MAX_OTHER}). Suspect misaligned class indexing."
        )
    if own < worst_val:
        warnings.append(
            f"C5's best class is '{worst_other}' ({worst_val:.4f}), not {SANITY_CLASS} "
            f"({own:.4f}). Column indexing is almost certainly wrong."
        )
    return warnings


def print_table(class_names: list[str], matrix: dict, stats: list[dict]) -> None:
    w = max(max(len(c) for c in class_names), 8) + 2
    rw = 30

    print("\n" + "=" * 72)
    print("COMPETENCE MATRIX -- per-class mAP50 on the centralized val set")
    print("rows = teacher (local model)   columns = class")
    print("=" * 72)
    header = f"{'':<{rw}s}" + "".join(f"{c:>{w}s}" for c in class_names)
    print(header)
    print("-" * len(header))
    for i in range(1, 7):
        key = f"local_c{i}"
        row = f"{key + '  (' + CLIENT_LABELS[i] + ')':<{rw}s}"
        row += "".join(f"{matrix[key][c]:>{w}.4f}" for c in class_names)
        print(row)
    print("-" * len(header))

    print(f"\n{'':<{rw}s}" + "".join(f"{c:>{w}s}" for c in class_names))
    for field, lab in (("argmax_teacher", "argmax teacher"),
                       ("top1", "top-1 mAP50"),
                       ("top2", "top-2 mAP50"),
                       ("margin", "margin (top1-top2)")):
        row = f"{lab:<{rw}s}"
        for s in stats:
            v = s[field]
            row += f"{v:>{w}s}" if isinstance(v, str) else f"{v:>{w}.4f}"
        print(row)

    n_distinct = len({s["argmax_teacher"] for s in stats})
    print(f"\ndistinct argmax teachers across the 6 columns: {n_distinct}/6")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute the 6x6 competence matrix.")
    ap.add_argument("--bank", default="experiments/teacher_bank/teacher_bank")
    ap.add_argument("--val_yaml", default="data/neu6_data/val/data.yaml")
    ap.add_argument("--out_dir", default="experiments/teacher_bank/competence")
    ap.add_argument("--device", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    bank, val_yaml = Path(args.bank), Path(args.val_yaml)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    class_names, matrix = score_bank(bank, val_yaml, args.device, args.imgsz, out_dir)
    stats = column_stats(class_names, matrix)
    print_table(class_names, matrix, stats)

    warnings = sanity_check(class_names, matrix)
    if warnings:
        print("\n" + "!" * 72)
        print("SANITY CHECK FAILED -- do not trust this matrix:")
        for w_ in warnings:
            print(f"  ! {w_}")
        print("!" * 72)
    else:
        print(f"\n[OK] Sanity: C5 (pitted-only) peaks on {SANITY_CLASS} "
              f"({matrix['local_c5'][SANITY_CLASS]:.4f}) and stays near zero elsewhere.")

    with open(out_dir / "competence_matrix.csv", "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["teacher", "label"] + class_names)
        for i in range(1, 7):
            k = f"local_c{i}"
            w_.writerow([k, CLIENT_LABELS[i]] + [f"{matrix[k][c]:.6f}" for c in class_names])

    (out_dir / "competence_matrix.json").write_text(json.dumps({
        "val_yaml": str(val_yaml.resolve()),
        "imgsz": args.imgsz,
        "class_names": class_names,
        "matrix": matrix,
        "per_class": stats,
        "sanity_warnings": warnings,
    }, indent=2))
    print(f"\n[DONE] -> {out_dir}/competence_matrix.{{csv,json}}")


if __name__ == "__main__":
    main()
