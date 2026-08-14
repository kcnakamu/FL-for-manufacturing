"""Post-disruption adaptation of the low-data backup client (C = client 2).

Module map:
  controller.py        - StageController: escalate head_only -> neck_head (-> full)
                         when validation improvement plateaus (pure logic, no torch)
  adaptive_finetune.py - driver: fine-tune in short segments, consulting the
                         controller between segments; writes trace.json
  kd.py                - class-retention-weighted output-level knowledge
                         distillation (teacher = pre-disruption global)
  distill_finetune.py  - driver: fine-tune C with the KD loss attached
"""

from .controller import Decision, StageController

__all__ = ["Decision", "StageController"]
