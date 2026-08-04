"""Non-invasive logging of the inputs Shapley reconstruction needs (spec 2.1).

Wraps a strategy's `aggregate_fit` exactly like strategies/delta_logging.py, so it
adds NO code to the training loop itself. Per federated round `t` it persists:

  * each participating client's full update omega_i^t  (the exact ndarrays the
    server aggregated -- so reconstruction reproduces the server model bit-for-bit),
  * each client's image count n_i (the FedAvg weight),
  * the aggregated GLOBAL model after round t,

plus, once at startup, the initial global (round 0) which serves as the pre-round
baseline / v(emptyset) for reconstruction.

On-disk layout (all under `log_dir`):

    manifest.json                      run-level: rule, disruption_round, format note
    globals/global_round_00.npz        initial global (baseline for v(emptyset))
    globals/global_round_01.npz        aggregated global AFTER round 1
    globals/global_round_02.npz        ...
    round_01/meta.json                 {round, rule, clients:[{cid,num_examples,file}]}
    round_01/client_0.npz              client 0's omega_i for round 1
    round_01/client_1.npz              ...
    ...

Arrays are stored as arr_0, arr_1, ... in model._state_keys() order -- the SAME
order get_parameters/set_parameters use -- so a loaded list can be fed straight
into reconstruct() and then set_parameters(). Reconstructing {A,B,C} at round t
reproduces globals/global_round_0t.npz (the real FedAvg result).

Reload with the helpers at the bottom: load_manifest / available_rounds /
load_round / load_global.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import parameters_to_ndarrays

NDArrays = List[np.ndarray]


# --------------------------------------------------------------------------- #
# low-level array (de)serialization
# --------------------------------------------------------------------------- #
def _save_ndarrays(path: Path, arrays: NDArrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"arr_{i}": a for i, a in enumerate(arrays)})


def _load_ndarrays(path: Path) -> NDArrays:
    with np.load(path, allow_pickle=False) as npz:
        return [npz[f"arr_{i}"] for i in range(len(npz.files))]


def _client_id(client_proxy, fit_res, idx: int) -> str:
    """Best-effort stable client label.

    The server-side ClientProxy.cid is a Flower-assigned node id, NOT the "0/1/2"
    the client was launched with -- so prefer the cid the client reports in its
    fit metrics (see client.fit). Fall back to the proxy cid, then to position.
    """
    metrics = getattr(fit_res, "metrics", None) or {}
    if "cid" in metrics and metrics["cid"] not in (None, ""):
        return str(metrics["cid"])
    cid = getattr(client_proxy, "cid", None)
    return str(cid) if cid is not None else str(idx)


# --------------------------------------------------------------------------- #
# the wrapper
# --------------------------------------------------------------------------- #
def add_update_logging(
    strategy,
    log_dir,
    rule: str = "fedavg",
    disruption_round: Optional[int] = None,
):
    """Wrap `strategy.aggregate_fit` to persist per-client updates + globals.

    Args:
        strategy:         any flwr strategy (compose after add_weight_delta_logging).
        log_dir:          directory to write the Shapley log tree into.
        rule:             aggregation rule name, recorded so reconstruct() can refuse
                          non-weighted-average rules (fedawa/adaptive/...).
        disruption_round: optional t* to tag in the manifest (analysis can also pick
                          it later, since every round is logged).
    """
    log_dir = Path(log_dir).resolve()
    (log_dir / "globals").mkdir(parents=True, exist_ok=True)

    # Record the initial (pre-round-1) global as the round-0 baseline for v(emptyset).
    seed = getattr(strategy, "initial_parameters", None)
    if seed is not None:
        try:
            _save_ndarrays(log_dir / "globals" / "global_round_00.npz",
                           parameters_to_ndarrays(seed))
        except Exception as e:  # never let logging break a run
            print(f"[ShapleyLog] WARNING: could not save initial global: {e}")

    manifest = {
        "rule": rule,
        "disruption_round": disruption_round,
        "baseline_global": "globals/global_round_00.npz",
        "format": ("npz-compressed; arrays keyed arr_0.. in model._state_keys() order; "
                   "reconstruct({A,B,C}) at round t == globals/global_round_<t:02d>.npz"),
        "note": ("v(emptyset) / pre-round baseline at disruption round t* is "
                 "globals/global_round_<t*-1:02d>.npz (the global broadcast INTO round t*)."),
    }
    with open(log_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    orig_aggregate_fit = strategy.aggregate_fit

    def aggregate_fit(server_round, results, failures):
        aggregated_parameters, metrics = orig_aggregate_fit(server_round, results, failures)
        try:
            round_dir = log_dir / f"round_{server_round:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            client_entries = []
            for idx, (client_proxy, fit_res) in enumerate(results):
                cid = _client_id(client_proxy, fit_res, idx)
                fname = f"client_{cid}.npz"
                arrays = parameters_to_ndarrays(fit_res.parameters)
                _save_ndarrays(round_dir / fname, arrays)
                client_entries.append({
                    "cid": cid,
                    "num_examples": int(fit_res.num_examples),
                    "n_arrays": len(arrays),
                    "file": fname,
                })

            with open(round_dir / "meta.json", "w") as fh:
                json.dump({"round": server_round, "rule": rule,
                           "clients": client_entries}, fh, indent=2)

            if aggregated_parameters is not None:
                _save_ndarrays(log_dir / "globals" / f"global_round_{server_round:02d}.npz",
                               parameters_to_ndarrays(aggregated_parameters))

            cids = ", ".join(e["cid"] for e in client_entries)
            print(f"[ShapleyLog] Round {server_round} | logged clients [{cids}] "
                  f"+ global -> {round_dir}")
        except Exception as e:
            print(f"[ShapleyLog] WARNING: round {server_round} logging failed: {e}")

        return aggregated_parameters, metrics

    strategy.aggregate_fit = aggregate_fit
    return strategy


# --------------------------------------------------------------------------- #
# reload helpers (used by evaluate.py / persistence.py)
# --------------------------------------------------------------------------- #
def load_manifest(log_dir) -> dict:
    with open(Path(log_dir) / "manifest.json") as fh:
        return json.load(fh)


def available_rounds(log_dir) -> List[int]:
    """Sorted list of federated rounds that have logged client updates."""
    log_dir = Path(log_dir)
    rounds = []
    for d in log_dir.glob("round_*"):
        if (d / "meta.json").exists():
            rounds.append(int(d.name.split("_")[1]))
    return sorted(rounds)


def load_round(log_dir, server_round: int) -> Tuple[Dict[str, NDArrays], Dict[str, int]]:
    """Return (updates_by_cid, counts_by_cid) for one round, ready for reconstruct()."""
    round_dir = Path(log_dir) / f"round_{server_round:02d}"
    with open(round_dir / "meta.json") as fh:
        meta = json.load(fh)
    updates: Dict[str, NDArrays] = {}
    counts: Dict[str, int] = {}
    for entry in meta["clients"]:
        cid = entry["cid"]
        arrays = _load_ndarrays(round_dir / entry["file"])
        if "n_arrays" in entry and len(arrays) != entry["n_arrays"]:
            raise ValueError(f"Round {server_round} client {cid}: expected "
                             f"{entry['n_arrays']} arrays, loaded {len(arrays)}.")
        updates[cid] = arrays
        counts[cid] = int(entry["num_examples"])
    return updates, counts


def load_global(log_dir, server_round: int) -> NDArrays:
    """Load the aggregated global after `server_round` (use 0 for the baseline)."""
    return _load_ndarrays(Path(log_dir) / "globals" / f"global_round_{server_round:02d}.npz")
