"""Adaptive stage selection: escalate the freeze regime on validation plateau.

Manuscript contribution 2: "adaptive head versus neck-and-head selection, based
on validation improvement rather than fixed rounds." The controller sees one
validation metric per training segment and decides whether to continue in the
current stage, escalate to the next (unfreezing more of the network), or stop.

Pure logic -- no torch / YOLO -- unit-tested in adaptation/tests/test_controller.py.
The driver (adaptive_finetune.py) maps stages onto Ultralytics runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

DEFAULT_STAGES = ["head_only", "neck_head"]


@dataclass
class Decision:
    action: str   # "continue" | "escalate" | "stop"
    mode: str     # stage active for the NEXT segment
    reason: str


class StageController:
    """Escalate when `patience` consecutive segments fail to beat the stage's
    best metric by more than `min_delta`; stop when the last stage plateaus.

    The stage-best resets on escalation so the new stage is judged against its
    own starting point, not the previous stage's peak.
    """

    def __init__(self, stages: Optional[List[str]] = None,
                 patience: int = 2, min_delta: float = 0.005):
        self.stages = list(DEFAULT_STAGES) if stages is None else list(stages)
        if not self.stages:
            raise ValueError("At least one stage is required.")
        if patience < 1:
            raise ValueError("patience must be >= 1.")
        self.patience = patience
        self.min_delta = min_delta
        self._stage_idx = 0
        self._best: Optional[float] = None
        self._stalls = 0

    @property
    def mode(self) -> str:
        return self.stages[self._stage_idx]

    def update(self, val_metric: float) -> Decision:
        """Consume one segment's validation metric; decide what to do next."""
        if self._best is None or val_metric > self._best + self.min_delta:
            self._best = max(val_metric, self._best or val_metric)
            self._stalls = 0
            return Decision("continue", self.mode,
                            f"improved to {val_metric:.4f} (stage best)")

        self._stalls += 1
        if self._stalls < self.patience:
            return Decision("continue", self.mode,
                            f"no improvement ({self._stalls}/{self.patience} stalls, "
                            f"best {self._best:.4f})")

        if self._stage_idx + 1 < len(self.stages):
            self._stage_idx += 1
            self._best = val_metric  # judge the new stage from here
            self._stalls = 0
            return Decision("escalate", self.mode,
                            f"plateaued after {self.patience} stalls -> unfreeze to "
                            f"'{self.mode}'")

        return Decision("stop", self.mode,
                        f"plateaued on final stage '{self.mode}'")
