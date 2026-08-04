"""
Collect per-round client metrics from an FL run and plot mAP50 across rounds.

Plots only mAP50. For all metrics (mAP50, mAP50-95, P, R, F1) in one figure,
use plot_fl_rounds.py instead.

Usage:
    # single run
    python utils/analysis/plot_metrics.py --run experiments/disruption_neu_fedavg/fl --save_fig map50.png

    # save DataFrame to CSV and figure to PNG
    python utils/analysis/plot_metrics.py --run experiments/disruption_neu_fedavg/fl --save_csv metrics.csv --save_fig map50.png

    # compare multiple runs on the same axes (one line per run × client)
    python utils/analysis/plot_metrics.py --run experiments/run_a/fl experiments/run_b/fl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from _fl_metrics import collect_metrics


def plot_map50(df: pd.DataFrame, title: str, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (run, client), group in df.groupby(["run", "client"], sort=True):
        label = f"client_{client}" if df["run"].nunique() == 1 else f"{run} / client_{client}"
        ax.plot(group["round"], group["mAP50"], marker="o", label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel("mAP50")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-client mAP50 across FL rounds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run", nargs="+", required=True,
        help="Path(s) to a run's fl/ directory (experiments/<exp_name>/fl)",
    )
    parser.add_argument("--save_csv", default=None, help="Save collected metrics to CSV")
    parser.add_argument("--save_fig", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    frames = []
    for run_path in args.run:
        run_dir = Path(run_path)
        if not run_dir.exists():
            sys.exit(f"Run directory not found: {run_dir}")
        df = collect_metrics(run_dir, include_run=True)
        if not df.empty:
            frames.append(df)

    if not frames:
        sys.exit("No metrics found in any of the provided run directories.")

    all_metrics = pd.concat(frames, ignore_index=True)
    print(all_metrics.to_string(index=False))

    if args.save_csv:
        all_metrics.to_csv(args.save_csv, index=False)
        print(f"CSV saved to {args.save_csv}")

    run_names = ", ".join(Path(r).name for r in args.run)
    plot_map50(all_metrics, title=f"mAP50 per client — {run_names}", save_path=args.save_fig)


if __name__ == "__main__":
    main()
