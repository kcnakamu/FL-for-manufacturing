"""Integration test: logger round-trip + reconstruction on real Flower objects.

Simulates one aggregate_fit call with three synthetic clients, then reloads the
logs and checks that reconstruct({A,B,C}) reproduces the logged global model
(spec acceptance test 5, end to end through the on-disk format).

Runnable:
    python -m pytest shapley/tests/test_logger.py
    python -m shapley.tests.test_logger
"""

from __future__ import annotations

import tempfile
from functools import reduce
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    _HAVE_FLWR = True
except Exception:  # pragma: no cover
    _HAVE_FLWR = False

from shapley.logger import (
    add_update_logging,
    available_rounds,
    load_global,
    load_manifest,
    load_round,
)
from shapley.reconstruct import reconstruct


def _model(seed, shapes=((4, 3), (5,), (2, 2))):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


def _fake_fitres(arrays, num_examples, cid):
    return SimpleNamespace(
        parameters=ndarrays_to_parameters(arrays),
        num_examples=num_examples,
        metrics={"cid": cid},
    )


def _make_strategy(initial):
    """Minimal strategy whose aggregate_fit does real FedAvg weighted averaging."""
    def aggregate_fit(server_round, results, failures):
        updates = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        counts = [fr.num_examples for _, fr in results]
        total = sum(counts)
        weighted = [[layer * n for layer in w] for w, n in zip(updates, counts)]
        agg = [reduce(np.add, layers) / total for layers in zip(*weighted)]
        return ndarrays_to_parameters(agg), {}

    return SimpleNamespace(
        initial_parameters=ndarrays_to_parameters(initial),
        aggregate_fit=aggregate_fit,
    )


def test_logger_roundtrip_and_reconstruction():
    if not _HAVE_FLWR:
        print("SKIP (flwr not installed)")
        return

    counts = {"0": 457, "1": 228, "2": 80}
    client_models = {"0": _model(1), "1": _model(2), "2": _model(3)}
    baseline = _model(0)

    strategy = _make_strategy(baseline)

    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "shapley_logs"
        strategy = add_update_logging(strategy, log_dir, rule="fedavg", disruption_round=4)

        # Deliberately shuffle order so identity must come from metrics["cid"].
        results = [
            (SimpleNamespace(cid="node-xyz"), _fake_fitres(client_models["2"], counts["2"], "2")),
            (SimpleNamespace(cid="node-abc"), _fake_fitres(client_models["0"], counts["0"], "0")),
            (SimpleNamespace(cid="node-def"), _fake_fitres(client_models["1"], counts["1"], "1")),
        ]
        agg_params, _ = strategy.aggregate_fit(1, results, [])
        logged_global = parameters_to_ndarrays(agg_params)

        # ---- manifest & structure ----
        manifest = load_manifest(log_dir)
        assert manifest["rule"] == "fedavg"
        assert manifest["disruption_round"] == 4
        assert available_rounds(log_dir) == [1]

        # ---- baseline (v(emptyset)) reloads to the initial global ----
        base_loaded = load_global(log_dir, 0)
        for g, b in zip(base_loaded, baseline):
            assert np.allclose(g, b)

        # ---- per-client updates reload correctly, keyed by launch cid ----
        updates, loaded_counts = load_round(log_dir, 1)
        assert set(updates) == {"0", "1", "2"}
        assert loaded_counts == counts
        for cid in updates:
            for a, b in zip(updates[cid], client_models[cid]):
                assert np.array_equal(a, b)

        # ---- reconstruct({A,B,C}) == the logged global (acceptance test 5) ----
        recon = reconstruct(updates, loaded_counts, ["0", "1", "2"], rule="fedavg")
        for r, g in zip(recon, logged_global):
            assert np.max(np.abs(r - g)) < 1e-6, np.max(np.abs(r - g))

        # ---- empty coalition uses the reloaded baseline ----
        recon_empty = reconstruct(updates, loaded_counts, [], baseline=base_loaded, rule="fedavg")
        for r, b in zip(recon_empty, baseline):
            assert np.allclose(r, b)


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} logger tests passed.")


if __name__ == "__main__":
    _run_all()
