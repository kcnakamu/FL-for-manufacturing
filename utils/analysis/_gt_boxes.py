"""Shared ground-truth label helpers for the utils/analysis scripts.

Imported as a sibling module (`from _gt_boxes import ...`) by scripts run as
`python utils/analysis/<script>.py`, matching the existing sibling-import
convention (see evaluate_nms.py importing evaluate_test.py).
"""
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_gt_boxes(label_path: Path):
    """Read a YOLO label file -> list of (class_id, cx, cy, w, h) in [0,1]."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append((cls, cx, cy, w, h))
    return boxes
