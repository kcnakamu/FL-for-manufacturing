"""Adaptive performance-weighted FedAvg.

Aggregate client updates weighted by validation performance (F1 + mAP50)
reported by each client in its fit() metrics.
"""
from functools import reduce

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


def _perf_score(metrics: dict) -> float:
    """score = 0.5 * F1 + 0.5 * mAP50, where F1 = 2PR / (P+R)."""
    p = float(metrics.get("precision", 0.0))
    r = float(metrics.get("recall", 0.0))
    m = float(metrics.get("mAP50", 0.0))
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return 0.5 * f1 + 0.5 * m


class AdaptiveWeightedFedAvg(fl.server.strategy.FedAvg):
    """Aggregate client updates weighted by validation performance (F1 + mAP50)."""

    def __init__(self, *args, eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        scores = [_perf_score(fit_res.metrics or {}) for _, fit_res in results]
        total = sum(scores)
        if total <= 0:
            weights = [1.0 / len(results)] * len(results)
            print(f"[Server] Round {server_round}: all perf scores zero — falling back to uniform weights")
        else:
            weights = [(s + self.eps) / (total + self.eps * len(results)) for s in scores]

        for (client_proxy, _), w, s in zip(results, weights, scores):
            cid = getattr(client_proxy, "cid", "?")
            print(f"[Server] Round {server_round} | client {cid} | score={s:.4f} | weight={w:.4f}")

        ndarrays_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        aggregated_ndarrays = [
            reduce(np.add, [w * layer for w, layer in zip(weights, layers)])
            for layers in zip(*ndarrays_list)
        ]
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        else:
            metrics_aggregated = {
                "mean_precision": float(np.mean([(fr.metrics or {}).get("precision", 0.0) for _, fr in results])),
                "mean_recall":    float(np.mean([(fr.metrics or {}).get("recall", 0.0)    for _, fr in results])),
                "mean_mAP50":     float(np.mean([(fr.metrics or {}).get("mAP50", 0.0)     for _, fr in results])),
                "mean_perf_score": float(np.mean(scores)),
            }
        return aggregated_parameters, metrics_aggregated


def build(shared_kwargs, eps: float = 1e-6, **_):
    return AdaptiveWeightedFedAvg(eps=eps, **shared_kwargs)
