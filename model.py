import random
import numpy as np
import torch
from ultralytics import YOLO
from collections import OrderedDict

MODEL_PATH = "yolov8n.pt"

# Optimization + augmentation shared by EVERY local training run in this repo --
# the standalone teacher bank and the federated clients alike.
#
# It lives here, in one place, because the two drifted apart once already and it
# was invisible: the teachers pinned SGD at lr0=0.01 while the clients passed
# nothing, so Ultralytics' optimizer="auto" silently resolved them to AdamW at
# lr0=0.001 (its auto rule picks AdamW with lr = 0.002*5/(4+nc) whenever a run is
# under 10k iterations, which every client round is). Teacher and student were
# training under a different optimizer AND a 10x different learning rate, which
# makes any comparison between them meaningless.
#
# Augmentation values are pinned rather than inherited so an Ultralytics upgrade
# cannot desynchronise runs that must stay comparable.
#
# Excluded on purpose: epochs, imgsz, batch and seed, which legitimately differ
# by role and are passed by the caller.
LOCAL_TRAIN_HP = {
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "deterministic": True,
    "amp": True,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5,
    "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5,
    "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
}


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs for reproducible model init.

    Controls the random detection-head initialization in load_model /
    DetectionModel. YOLO's own training randomness (augmentation, dataloader
    order) is seeded separately by passing seed=<seed> to model.train().
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(num_classes=6):
    """Load COCO-pretrained YOLOv8n and adapt the head to num_classes.
    """
    model = YOLO(MODEL_PATH)
    if model.model.nc != num_classes:
        from ultralytics.nn.tasks import DetectionModel

        pretrained_state = model.model.state_dict()
        new_model = DetectionModel(model.model.yaml, nc=num_classes).to(model.device)
        new_state = new_model.state_dict()

        # Keep pretrained weights wherever the tensor exists and shapes match.
        transfer = {
            k: v
            for k, v in pretrained_state.items()
            if k in new_state and v.shape == new_state[k].shape
        }
        missing = len(new_state) - len(transfer)
        new_model.load_state_dict(transfer, strict=False)
        print(
            f"[load_model] Adapted head to nc={num_classes}: "
            f"transferred {len(transfer)} pretrained tensors, "
            f"{missing} randomly initialized (head)."
        )

        new_model.nc = num_classes
        model.model = new_model
    return model

# YOLOv8n layer indices: backbone 0-9, neck 10-21, head 22.
# https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers
_BACKBONE_END = 10
_NECK_END = 22


def freeze_indices(mode: str) -> list:
    """Layer indices to pass as model.train(freeze=...) for staged fine-tuning.

    Ultralytics' BaseTrainer force-re-enables requires_grad for any float
    param not covered by the `freeze` train arg, so setting requires_grad
    manually (apply_freeze) is NOT sufficient — the freeze list must go
    through .train(freeze=freeze_indices(mode)).

    Modes: 'head_only' (freeze backbone+neck), 'neck_head' (freeze backbone),
    'full' (train everything).
    """
    if mode == "head_only":
        return list(range(_NECK_END))
    if mode == "neck_head":
        return list(range(_BACKBONE_END))
    return []


def apply_freeze(model, mode: str) -> None:
    """Freeze layers for staged fine-tuning.

    Shared by scripts/train_centralized.py and shapley/persistence.py so the
    backbone/neck/head boundaries live in exactly one place.

    NOTE: for training, callers must ALSO pass freeze=freeze_indices(mode) to
    model.train() — the Ultralytics trainer re-enables requires_grad for any
    param not listed there, silently undoing this function.

    Modes: 'head_only' (freeze backbone+neck), 'neck_head' (freeze backbone),
    'full' (train everything).
    """
    frozen = set(freeze_indices(mode))

    for i, layer in enumerate(model.model.model):
        for param in layer.parameters():
            param.requires_grad = i not in frozen

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"Mode '{mode}': {trainable:,} / {total:,} trainable ({100 * trainable / total:.1f}%)")


def _state_keys(model):
    """Ordered state_dict keys — the canonical tensor order shared across clients."""
    return list(model.model.state_dict().keys())

def get_parameters(model):
    """Share ALL model tensors as a list of numpy arrays, in _state_keys() order."""
    state = model.model.state_dict()
    keys = _state_keys(model)
    return [state[k].detach().cpu().numpy().copy() for k in keys]

def set_parameters(model, parameters):
    """Load global weights (list of numpy arrays in _state_keys() order) into the model."""
    current_state = model.model.state_dict()
    keys = _state_keys(model)

    if len(parameters) != len(keys):
        raise ValueError(
            f"Parameter count mismatch: got {len(parameters)} arrays, expected {len(keys)}. "
            "Ensure all clients and server are using the same model architecture."
        )

    updated_state = OrderedDict()
    for k, v in zip(keys, parameters):
        src = torch.from_numpy(v.copy())
        ref = current_state[k]
        
        if tuple(src.shape) != tuple(ref.shape):
            raise ValueError(
                f"Shape mismatch for {k}: got {tuple(src.shape)}, expected {tuple(ref.shape)}."
            )
        updated_state[k] = src.to(device=ref.device, dtype=ref.dtype)

    # Load the updated state. strict=False is used to ignore non-persistent buffers 
    # if they weren't included in the parameter list.
    model.model.load_state_dict(updated_state, strict=False)

