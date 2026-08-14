"""Tests for the adaptive stage controller (no torch / YOLO).

Runnable:
    python -m pytest adaptation/tests/test_controller.py
    python -m adaptation.tests.test_controller
"""

from __future__ import annotations

from adaptation.controller import StageController


def _feed(ctrl, values):
    return [ctrl.update(v) for v in values]


def test_improving_sequence_never_escalates():
    ctrl = StageController(patience=2, min_delta=0.005)
    decisions = _feed(ctrl, [0.1, 0.2, 0.3, 0.4])
    assert all(d.action == "continue" for d in decisions)
    assert ctrl.mode == "head_only"


def test_flat_sequence_escalates_after_exactly_patience():
    ctrl = StageController(patience=2, min_delta=0.005)
    d = _feed(ctrl, [0.3, 0.3, 0.3])
    assert [x.action for x in d] == ["continue", "continue", "escalate"]
    assert d[2].mode == "neck_head"


def test_boundary_equal_to_min_delta_is_a_stall():
    # val == best + min_delta must NOT count as improvement (strict >).
    ctrl = StageController(patience=1, min_delta=0.005)
    first = ctrl.update(0.300)
    assert first.action == "continue"
    second = ctrl.update(0.305)
    assert second.action == "escalate"


def test_last_stage_plateau_stops():
    ctrl = StageController(stages=["head_only", "neck_head"], patience=1)
    assert ctrl.update(0.3).action == "continue"
    assert ctrl.update(0.3).action == "escalate"   # -> neck_head, stage best resets to 0.3
    # Another flat 0.3 is a stall; patience=1 on the final stage -> stop.
    assert ctrl.update(0.3).action == "stop"
    assert ctrl.mode == "neck_head"


def test_escalation_resets_stall_counter_and_stage_best():
    ctrl = StageController(stages=["head_only", "neck_head"], patience=2)
    _feed(ctrl, [0.5, 0.5, 0.5])          # escalate at third call
    assert ctrl.mode == "neck_head"
    # New stage improves over its own baseline (0.5) -> continue, twice.
    d = _feed(ctrl, [0.52, 0.54])
    assert [x.action for x in d] == ["continue", "continue"]
    # Then two stalls on the FINAL stage -> stop (not escalate).
    d = _feed(ctrl, [0.54, 0.54])
    assert [x.action for x in d] == ["continue", "stop"]


def test_three_stage_escalation_chain():
    ctrl = StageController(stages=["head_only", "neck_head", "full"], patience=1)
    assert ctrl.update(0.3).action == "continue"
    assert ctrl.update(0.3).action == "escalate"
    assert ctrl.mode == "neck_head"
    assert ctrl.update(0.3).action == "escalate"
    assert ctrl.mode == "full"
    assert ctrl.update(0.3).action == "stop"
    assert ctrl.mode == "full"


def test_single_stage_plateau_stops_directly():
    ctrl = StageController(stages=["neck_head"], patience=2)
    d = _feed(ctrl, [0.4, 0.4, 0.4])
    assert [x.action for x in d] == ["continue", "continue", "stop"]


def test_stage_best_tracks_maximum_not_latest():
    # A dip below the stage best must not reset the bar.
    ctrl = StageController(patience=3, min_delta=0.005)
    _feed(ctrl, [0.5])
    d = _feed(ctrl, [0.4, 0.49, 0.502])  # none beat 0.5 + 0.005
    assert [x.action for x in d] == ["continue", "continue", "escalate"]


def test_invalid_args_raise():
    for bad in (dict(stages=[]), dict(patience=0)):
        try:
            StageController(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} controller tests passed.")


if __name__ == "__main__":
    _run_all()
