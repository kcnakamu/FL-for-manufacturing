"""Federated aggregation strategies, one per file.

To add a strategy: create `strategies/<name>.py` exposing a
`build(shared_kwargs, **params)` function, then register it in BUILDERS below.
"""
from . import fedavg, fedprox, adaptive, fedawa
from .delta_logging import add_weight_delta_logging

# name -> build(shared_kwargs, **params) -> flwr strategy instance
BUILDERS = {
    "fedavg":   fedavg.build,
    "fedprox":  fedprox.build,
    "adaptive": adaptive.build,
    "fedawa":   fedawa.build,
}

STRATEGIES = list(BUILDERS)


def build_strategy(name, shared_kwargs, **params):
    """Construct a strategy by name. Extra `params` (mu, temperature, eps, ...)
    are forwarded to the builder, which takes what it needs and ignores the rest.
    """
    if name not in BUILDERS:
        raise ValueError(f"Unknown strategy {name!r}; choose from {STRATEGIES}")
    return BUILDERS[name](shared_kwargs, **params)


__all__ = ["build_strategy", "add_weight_delta_logging", "STRATEGIES", "BUILDERS"]
