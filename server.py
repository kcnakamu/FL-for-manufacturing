import flwr as fl
import argparse
from model import get_parameters, set_parameters, load_model


def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    return {
        "mAP50":    sum(n * m["mAP50"]    for n, m in metrics) / total,
        "mAP50-95": sum(n * m["mAP50-95"] for n, m in metrics) / total,
    } if total > 0 else {}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=1)
    args = parser.parse_args()

    print("Loading model")
    server_model = load_model(num_classes=args.num_classes)
    
    class BackboneStrategy(fl.server.strategy.FedAvg):
        def initialize_parameters(self, client_manager):
            """Initialize global model with backbone-only parameters."""
            initial_params = get_parameters(server_model)
            return fl.common.ndarrays_to_parameters(initial_params)
        
        def aggregate_fit(self, server_round, results, failures):
            # Check all clients sent correct param count
            param_counts = [len(fl.common.parameters_to_ndarrays(fit_res.parameters)) 
                            for _, fit_res in results]
            print(f"[Server] Round {server_round}: Received param counts from clients: {param_counts}")
            # print(f"Param keys: {results[0].parameters}")
            
            aggregated = super().aggregate_fit(server_round, results, failures)
            
            if aggregated is not None:
                agg_params = fl.common.parameters_to_ndarrays(aggregated[0])
                print(f"[Server] Round {server_round}: Sending {len(agg_params)} params to clients")
                set_parameters(server_model, agg_params)
            
            return aggregated

    initial_params = fl.common.ndarrays_to_parameters(get_parameters(server_model))

    print("Setting strategy")
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1.0,
    #     min_fit_clients=3,
    #     min_available_clients=3,
    #     initial_parameters=initial_params,
    #     evaluate_metrics_aggregation_fn=weighted_average,
    # )
    # strategy = fl.server.strategy.FedProx(
    #     fraction_fit=1.0,
    #     min_fit_clients=3,
    #     min_available_clients=3,
    #     initial_parameters=initial_params,
    #     evaluate_metrics_aggregation_fn=weighted_average,
    #     proximal_mu=1.0,  
    # )
    strategy = BackboneStrategy(
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
