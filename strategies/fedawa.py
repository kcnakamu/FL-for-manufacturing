"""FedAWA: aggregate by cosine similarity of weight deltas.

Per-client score = mean cosine sim of its delta (ΔW_i = W_local_i - W_global)
to all other clients' deltas. Aggregation weights = softmax(scores / temperature).
"""
from functools import reduce

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class FedAWA(fl.server.strategy.FedAvg):
    """Aggregate by cosine similarity of weight deltas (ΔW_i = W_local_i - W_global)."""

    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self._global_ndarrays = None

    def configure_fit(self, server_round, parameters, client_manager):
        # Cache the global weights so aggregate_fit can compute deltas.
        self._global_ndarrays = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Compute ΔW_i = W_local_i − W_global (layer-wise), then flatten.
        if self._global_ndarrays is not None:
            flat_deltas = [
                np.concatenate([(l - g).flatten() for l, g in zip(local, self._global_ndarrays)])
                for local in client_ndarrays
            ]
        else:
            # Round 0 fallback: no global weights cached yet; use raw weights as proxy.
            flat_deltas = [
                np.concatenate([layer.flatten() for layer in local])
                for local in client_ndarrays
            ]

        n = len(flat_deltas)
        norms = [np.linalg.norm(d) for d in flat_deltas]

        # Pairwise cosine similarity matrix (diagonal = 0).
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if norms[i] > 0 and norms[j] > 0:
                    s = np.dot(flat_deltas[i], flat_deltas[j]) / (norms[i] * norms[j])
                else:
                    s = 0.0
                sim_matrix[i, j] = s
                sim_matrix[j, i] = s

        # Per-client score = mean cosine sim to all other clients.
        scores = sim_matrix.sum(axis=1) / max(n - 1, 1)

        # Softmax with temperature; shift for numerical stability.
        shifted = scores - scores.max()
        exp_s = np.exp(shifted / self.temperature)
        weights = exp_s / exp_s.sum()

        for (client_proxy, _), w, s, norm in zip(results, weights, scores, norms):
            cid = getattr(client_proxy, "cid", "?")
            print(
                f"[FedAWA] Round {server_round} | client {cid} "
                f"| Δnorm={norm:.4f} | cos_score={s:.4f} | weight={w:.4f}"
            )

        aggregated_ndarrays = [
            reduce(np.add, [w * layer for w, layer in zip(weights, layers)])
            for layers in zip(*client_ndarrays)
        ]
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {"mean_cosine_score": float(np.mean(scores))}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))

        return aggregated_parameters, metrics_aggregated


def build(shared_kwargs, temperature: float = 1.0, **_):
    return FedAWA(temperature=temperature, **shared_kwargs)
