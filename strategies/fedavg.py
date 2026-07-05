"""Plain FedAvg — the stock Flower strategy, no custom code."""
import flwr as fl


def build(shared_kwargs, **_):
    return fl.server.strategy.FedAvg(**shared_kwargs)
