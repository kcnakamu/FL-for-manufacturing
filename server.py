import flwr as fl
import argparse
import numpy as np
from functools import reduce
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from model import get_parameters, load_model

STRATEGIES = ["adaptive", "fedawa", "fedavg", "fedprox"]

def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    return {
        "mAP50":    sum(n * m["mAP50"]    for n, m in metrics) / total,
        "mAP50-95": sum(n * m["mAP50-95"] for n, m in metrics) / total,
    } if total > 0 else {}

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



class FedAWA(fl.server.strategy.FedAvg):
    """Aggregate by cosine similarity of weight deltas (ΔW_i = W_local_i - W_global).

    Per-client score = mean cosine sim of its delta to all other clients' deltas.
    Aggregation weights = softmax(scores / temperature).
    """

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="fedavg",
        help="Aggregation strategy: adaptive, fedawa, fedavg, or fedprox",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="Proximal term coefficient for FedProx",
    )
    args = parser.parse_args()

    print("Loading model")

    server_model = load_model(num_classes=args.num_classes)
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(server_model))

    shared_kwargs = dict(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Setting strategy: {args.strategy}")
    if args.strategy == "fedawa":
        strategy = FedAWA(**shared_kwargs)
    elif args.strategy == "fedavg":
        strategy = fl.server.strategy.FedAvg(**shared_kwargs)
    elif args.strategy == "fedprox":
        strategy = fl.server.strategy.FedProx(proximal_mu=args.mu, **shared_kwargs)
    else:  # "adaptive"
        strategy = AdaptiveWeightedFedAvg(**shared_kwargs)

    print("Starting server")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()
