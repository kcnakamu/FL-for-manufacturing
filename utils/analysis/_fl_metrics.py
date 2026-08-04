"""Shared collector for per-round FL client metrics.

Imported as a sibling module (`from _fl_metrics import collect_metrics`) by the
plotting scripts run as `python utils/analysis/<script>.py`.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd


def collect_metrics(run_dir: Path, include_run: bool = False) -> pd.DataFrame:
    """Walk <run_dir>/round_*/client_*_val/metrics.json and return a DataFrame.

    When include_run=True, each row is prefixed with a "run" column (the run
    dir name) so several runs can be concatenated and compared; when False the
    frame has no "run" column. Column order is preserved for both callers.
    """
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
        row = {"round": int(round_match.group(1)), "client": client_match.group(1), **data}
        if include_run:
            row = {"run": run_dir.name, **row}
        rows.append(row)

    if not rows:
        print(f"Warning: no metrics.json files found under {run_dir}", file=sys.stderr)
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["client", "round"]).reset_index(drop=True)
