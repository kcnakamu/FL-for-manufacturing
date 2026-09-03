import flwr as fl
import argparse
from model import get_parameters, load_model, set_seed
from strategies import build_strategy, add_weight_delta_logging, STRATEGIES
from shapley.logger import add_update_logging


def weighted_average(metrics):
    """Evaluate-metric aggregation, weighted by client num_examples."""
    total = sum(n for n, _ in metrics)
    return {
        "mAP50":    sum(n * m["mAP50"]    for n, m in metrics) / total,
        "mAP50-95": sum(n * m["mAP50-95"] for n, m in metrics) / total,
    } if total > 0 else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--port", type=int, default=8080,
                        help="gRPC port. MUST be unique per concurrent run: if two "
                             "jobs land on the same node and share a port, only one "
                             "server binds and every client from BOTH jobs connects "
                             "to it, silently aggregating across runs.")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--num_clients", type=int, default=6,
                        help="Clients required before a round starts. Must equal "
                             "the number actually launched: if it is lower, Flower "
                             "begins as soon as that many connect and silently "
                             "trains on a subset.")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="fedavg",
        help="Aggregation strategy: " + ", ".join(STRATEGIES),
    )
    parser.add_argument("--mu", type=float, default=0.01,
                        help="Proximal term coefficient for FedProx")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for FedAWA")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Smoothing epsilon for adaptive weighting")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the initial global model (head init) "
                             "broadcast to all clients. Vary across runs to test "
                             "robustness to initialization.")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="If set, persist per-round client updates + global "
                             "checkpoints here for Shapley contribution analysis "
                             "(see shapley/logger.py). Omit to disable logging.")
    parser.add_argument("--disruption_round", type=int, default=None,
                        help="Optional t* to tag in the Shapley log manifest (the "
                             "round A/B go offline). Every round is logged regardless.")
    args = parser.parse_args()

    # Seed before building the model so the randomly initialized detection head
    # (broadcast to every client as the round-1 global weights) is reproducible.
    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    print("Loading model")
    server_model = load_model(num_classes=args.num_classes)
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(server_model))

    shared_kwargs = dict(
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Waiting for {args.num_clients} clients per round on port {args.port}")
    print(f"Setting strategy: {args.strategy}")
    strategy = build_strategy(
        args.strategy, shared_kwargs,
        mu=args.mu, temperature=args.temperature, eps=args.eps,
    )
    strategy = add_weight_delta_logging(strategy)

    if args.log_dir:
        print(f"Shapley update logging -> {args.log_dir}")
        strategy = add_update_logging(
            strategy, args.log_dir,
            rule=args.strategy, disruption_round=args.disruption_round,
        )

    print("Starting server")
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()
