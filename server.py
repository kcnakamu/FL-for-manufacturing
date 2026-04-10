import flwr as fl
import argparse
from model import load_model, get_parameters


def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    return {
        "mAP50":    sum(n * m["mAP50"]    for n, m in metrics) / total,
        "mAP50-95": sum(n * m["mAP50-95"] for n, m in metrics) / total,
    } if total > 0 else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    print("Loading model")
    model = load_model(num_classes=6)
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(model))

    print("Setting strategy")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    print("Starting server")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()