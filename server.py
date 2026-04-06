import flwr as fl


def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    return {
        "mAP50":    sum(n * m["mAP50"]    for n, m in metrics) / total,
        "mAP50-95": sum(n * m["mAP50-95"] for n, m in metrics) / total,
    } if total > 0 else {}


def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=30),
    )

if __name__ == "__main__":
    main()