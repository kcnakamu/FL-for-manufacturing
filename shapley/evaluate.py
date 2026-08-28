"""Evaluator: v(S) = mAP50 of a saved model on the shared test set (spec 2.3).

Every model the driver evaluates -- a reconstructed coalition (saved to .pt) or a
fine-tuning checkpoint -- is a .pt file, so there is a single evaluation entry
point: `evaluate_checkpoint`. It returns the scalar utility (mAP50) plus per-class
AP (for the section-3 forgetting proxy).

Mirrors the val pattern proven in utils/analysis/evaluate_test.py: run
YOLO.val(split="test") on the shared test set and read metrics.box.*. The test
data.yaml is built once by the caller via `build_test_yaml` and reused across all
evaluations (it is constant for a whole analysis).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Default test imgsz matches client training/eval (imgsz=480) and evaluate_test.py.
_DEFAULT_IMGSZ = 480


def build_test_yaml(test_dir, class_names: List[str]) -> str:
    """Write a temp data.yaml pointing at the shared test set; return its path.

    Same shape as utils/analysis/evaluate_test.build_test_yaml -- kept local so
    this module has no cross-package import dependency. Build once per analysis
    and pass the path to every evaluate_checkpoint call.
    """
    test_dir = Path(test_dir)
    data = {
        "path": str(test_dir.resolve()),
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": len(class_names),
        "names": list(class_names),
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                      prefix="shapley_test_")
    yaml.dump(data, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return tmp.name


def _device(device: Optional[str]) -> str:
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_checkpoint(
    pt_path,
    test_yaml: str,
    device: Optional[str] = None,
    imgsz: int = _DEFAULT_IMGSZ,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    out_dir: str = "shapley_eval",
    name: str = "checkpoint",
    split: str = "test",
) -> Dict:
    """v(S) for a saved .pt model: mAP50, mAP50-95, and per-class AP@50.

    `split` selects which key of the data yaml to evaluate ("test" by default;
    pass "val" to score against a centralized validation set).
    """
    from ultralytics import YOLO

    device = _device(device)
    model = YOLO(str(pt_path))

    # Ultralytics uses iou=0.7 for NMS by default; only override when supplied
    # (matches utils/analysis/evaluate_test.py).
    val_kwargs = {}
    if iou is not None:
        val_kwargs["iou"] = iou

    metrics = model.val(
        data=test_yaml,
        split=split,
        imgsz=imgsz,
        batch=16,
        workers=0,
        verbose=False,
        device=device,
        conf=conf,
        project=out_dir,
        name=name,
        exist_ok=True,
        **val_kwargs,
    )

    # Per-class AP@50: ap_class_index lists only classes present in the test set,
    # so index the per-class arrays through it (never assume 0..nc-1).
    # Class names come from the test yaml, NOT model.names: models rebuilt from
    # logged weights (persistence._save_pt) carry default names ('0','1','2'),
    # which would silently mismatch the real class names downstream
    # (contribution matrix, per-class forgetting).
    with open(test_yaml) as fh:
        names = yaml.safe_load(fh)["names"]
    per_class = {str(names[int(c)]): float(metrics.box.ap50[i])
                 for i, c in enumerate(metrics.box.ap_class_index)}

    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "per_class_ap50": per_class,
    }
