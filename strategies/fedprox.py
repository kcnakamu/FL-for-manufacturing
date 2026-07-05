"""FedProx — the stock Flower strategy with a proximal term.

The proximal term itself is applied client-side (see client._fedprox_hook);
the server just forwards `proximal_mu` to clients via the fit config.
"""
import flwr as fl


def build(shared_kwargs, mu: float = 0.01, **_):
    return fl.server.strategy.FedProx(proximal_mu=mu, **shared_kwargs)
