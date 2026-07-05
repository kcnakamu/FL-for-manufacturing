"""Cross-cutting weight-delta logging — wraps any strategy's aggregate_fit.

Logs the L2 norm of the change in the global aggregated weights between
consecutive rounds. Near-zero after round ~2-3 → genuine convergence.
Consistently large while eval metrics stay flat → aggregation / weight-loading bug.
"""
import numpy as np
from flwr.common import parameters_to_ndarrays


def _flatten_float(ndarrays):
    """Flatten only floating-point tensors into one vector.

    Integer buffers (e.g. BatchNorm num_batches_tracked) are excluded so the
    norm reflects actual weight movement, not per-round batch counters.
    """
    floats = [a.astype(np.float64).flatten() for a in ndarrays
              if np.issubdtype(a.dtype, np.floating)]
    return np.concatenate(floats) if floats else np.array([])


def add_weight_delta_logging(strategy):
    """Wrap strategy.aggregate_fit to log ||Δglobal||₂ between rounds."""
    orig_aggregate_fit = strategy.aggregate_fit
    # Seed the baseline with the initial (pretrained) global weights so the
    # round-1 delta tells you how far round 1 moved from the pretrained init.
    seed = getattr(strategy, "initial_parameters", None)
    state = {"prev": _flatten_float(parameters_to_ndarrays(seed)) if seed is not None else None}

    def aggregate_fit(server_round, results, failures):
        aggregated_parameters, metrics = orig_aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            flat_new = _flatten_float(parameters_to_ndarrays(aggregated_parameters))
            prev = state["prev"]
            if prev is not None and prev.size == flat_new.size:
                delta = float(np.linalg.norm(flat_new - prev))
                rel = delta / (float(np.linalg.norm(prev)) + 1e-12)
                print(f"[WeightDelta] Round {server_round} | "
                      f"||Δglobal||₂={delta:.6f} | relative={rel:.6e}")
                metrics = {**(metrics or {}),
                           "global_weight_delta_l2": delta,
                           "global_weight_delta_rel": rel}
            else:
                print(f"[WeightDelta] Round {server_round} | baseline established")
            state["prev"] = flat_new
        return aggregated_parameters, metrics

    strategy.aggregate_fit = aggregate_fit
    return strategy
