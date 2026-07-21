import random
import numpy as np
import torch
from ultralytics import YOLO
from collections import OrderedDict

MODEL_PATH = "yolov8n.pt"


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

def load_model(num_classes=1):
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

