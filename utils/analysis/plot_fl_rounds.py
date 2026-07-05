"""
Collect per-round, per-client metrics from an FL run and plot the performance
trend across all rounds.

Walks <run_dir>/round_*/client_*_val/metrics.json, builds a tidy DataFrame, and
draws one subplot per metric (mAP50, mAP50-95, precision, recall, f1) with one
line per client. Optionally dumps the collected metrics to CSV.

Usage:
    # point at the fl/ folder of a run
    python utils/analysis/plot_fl_rounds.py --run experiments/disruption_neu_fedavg/fl

    # also save the figure and the raw table
    python utils/analysis/plot_fl_rounds.py \
        --run experiments/disruption_neu_fedavg/fl \
        --save_fig experiments/disruption_neu_fedavg/fl/round_trends.png \
        --save_csv experiments/disruption_neu_fedavg/fl/round_metrics.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Metrics to plot, in display order. Only those present in the data are drawn.
METRICS = ["mAP50", "mAP50-95", "precision", "recall", "f1"]


def collect_metrics(run_dir: Path) -> pd.DataFrame:
    """Walk <run_dir>/round_*/client_*_val/metrics.json and return a DataFrame."""
    round_pattern = re.compile(r"round_(\d+)")
    client_pattern = re.compile(r"client_(\w+)_val")

    rows = []
    for metrics_file in sorted(run_dir.glob("round_*/client_*_val/metrics.json")):
        round_match = round_pattern.search(metrics_file.parts[-3])
        client_match = client_pattern.search(metrics_file.parts[-2])
        if not round_match or not client_match:
            continue
        with open(metrics_file) as f:
            data = json.load(f)
        rows.append({
            "round":  int(round_match.group(1)),
            "client": client_match.group(1),
            **data,
        })

    if not rows:
        print(f"Warning: no metrics.json files found under {run_dir}", file=sys.stderr)
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["client", "round"]).reset_index(drop=True)


def plot_trends(df: pd.DataFrame, title: str, save_path: str | None = None):
    metrics = [m for m in METRICS if m in df.columns]
    if not metrics:
        sys.exit(f"None of the expected metrics {METRICS} found in the data.")

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    clients = sorted(df["client"].unique())
    for ax, metric in zip(axes, metrics):
        for client in clients:
            group = df[df["client"] == client].sort_values("round")
            ax.plot(group["round"], group[metric], marker="o", markersize=3, label=f"client_{client}")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    # Hide any unused subplots.
    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-client metric trends across FL rounds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True, help="Path to the run's fl/ directory")
    parser.add_argument("--save_csv", default=None, help="Save collected metrics to CSV")
    parser.add_argument("--save_fig", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    df = collect_metrics(run_dir)
    if df.empty:
        sys.exit("No metrics found.")

    print(df.to_string(index=False))
    print(f"\nRounds: {df['round'].min()}–{df['round'].max()} "
          f"({df['round'].nunique()} rounds, {sorted(df['client'].unique())} clients)")

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"CSV saved to {args.save_csv}")

    plot_trends(df, title=f"FL metric trends — {run_dir.parent.name}", save_path=args.save_fig)


if __name__ == "__main__":
    main()
