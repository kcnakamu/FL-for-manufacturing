"""Exact Shapley-based contribution persistence for the 3-client disruption experiment.

Players are the 3 clients (A=client_0, B=client_1, C=client_2), so there are only
2**3 = 8 coalitions and Shapley is computed EXACTLY (no Monte-Carlo sampling).

Module map (see docs/shapley_spec.md):
  reconstruct.py  - rebuild a coalition's aggregated model from logged per-client updates
  shapley.py      - exact N=3 Shapley from the 8 coalition utilities (+ general fallback)
  evaluate.py     - v(S) = mAP50 of a reconstructed model on the shared test set
  logger.py       - non-invasive capture of per-round client updates + global checkpoints
  persistence.py  - driver: retention curve rho_i(tau) + per-class forgetting proxy
"""

from .shapley import exact_shapley, general_shapley, coalitions
from .reconstruct import reconstruct, AGGREGATORS, UnsupportedAggregatorError

__all__ = [
    "exact_shapley",
    "general_shapley",
    "coalitions",
    "reconstruct",
    "AGGREGATORS",
    "UnsupportedAggregatorError",
]
