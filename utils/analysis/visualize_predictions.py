"""
Render ground-truth vs. predicted bounding boxes for the test set into one PDF.

Each page shows a single test image twice, side by side:
    left  = ground-truth boxes (from the YOLO label file)
    right = model predictions (with confidence scores)

so you can scroll through the whole test set in one document instead of
opening images one at a time.

Usage:
    python utils/analysis/visualize_predictions.py \
        --model experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
        --data_dir data/neu_data --class_names Inclusion Patches Scratches

    # custom confidence threshold and output path
    python utils/analysis/visualize_predictions.py --model best.pt \
        --data_dir data/neu_data --conf 0.25 --output predictions.pdf
"""

import argparse
import sys
from pathlib import Path

from _gt_boxes import IMAGE_EXTS, load_gt_boxes

# Distinct colors per class index; ground truth and predictions share the
# palette so a green GT box and a green pred box are the same class.
CLASS_COLORS = [
    "#2ca02c",  # green
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
]


def draw_box(ax, x1, y1, x2, y2, color, label):
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle((x1, y1), x2 - x1, y2 - y1,
                  fill=False, edgecolor=color, linewidth=2)
    )
    ax.text(
        x1, y1 - 4, label, color="white", fontsize=8, va="bottom",
        bbox=dict(facecolor=color, edgecolor="none", pad=1, alpha=0.8),
    )


def class_color(cls: int) -> str:
    return CLASS_COLORS[cls % len(CLASS_COLORS)]


def render_page(pdf, img_path, gt_boxes, result, class_names):
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax in (ax_gt, ax_pred):
        ax.imshow(img)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

    # Ground truth (normalized xywh -> pixel corners)
    for cls, cx, cy, w, h in gt_boxes:
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        name = class_names[cls] if cls < len(class_names) else str(cls)
        draw_box(ax_gt, x1, y1, x2, y2, class_color(cls), name)

    # Predictions (Ultralytics already gives pixel xyxy)
    n_pred = 0
    if result is not None and result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = class_names[cls] if cls < len(class_names) else str(cls)
            draw_box(ax_pred, x1, y1, x2, y2, class_color(cls),
                     f"{name} {conf:.2f}")
            n_pred += 1

    ax_gt.set_title(f"Ground truth ({len(gt_boxes)} boxes)", fontsize=11)
    ax_pred.set_title(f"Predictions ({n_pred} boxes)", fontsize=11)
    fig.suptitle(img_path.name, fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Render GT vs. predicted boxes for the test set into a PDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Path to a .pt weight file")
    parser.add_argument("--data_dir", default="data",
                        help="Root data directory containing the 'test/' subfolder")
    parser.add_argument("--class_names", nargs="+",
                        default=["Inclusion", "Patches", "Scratches"],
                        help="YOLO class names (order must match label class IDs)")
    parser.add_argument("--output", default="predictions.pdf",
                        help="Output PDF path")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="Inference image size (match training/eval)")
    parser.add_argument("--device", default=None,
                        help="Device override, e.g. 'cpu', '0' (auto-detected if omitted)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only render the first N images (for a quick look)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import torch
    from matplotlib.backends.backend_pdf import PdfPages
    from ultralytics import YOLO

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = Path(args.data_dir) / "test"
    img_dir = test_dir / "images"
    label_dir = test_dir / "labels"
    if not img_dir.exists():
        sys.exit(f"Test images directory not found: {img_dir}")

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {img_dir}")
    if args.limit is not None:
        images = images[:args.limit]

    print(f"Test set: {img_dir.resolve()}")
    print(f"Model:    {args.model}")
    print(f"Device:   {device}")
    print(f"Images:   {len(images)}  (conf={args.conf})\n")

    model = YOLO(args.model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        for i, img_path in enumerate(images):
            gt_boxes = load_gt_boxes(label_dir / f"{img_path.stem}.txt")
            result = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )[0]
            render_page(pdf, img_path, gt_boxes, result, args.class_names)
            print(f"[{i+1}/{len(images)}] {img_path.name}")

    print(f"\nSaved {len(images)}-page PDF to {out_path.resolve()}")


if __name__ == "__main__":
    main()
