import torch
from ultralytics import YOLO
from collections import OrderedDict

MODEL_PATH = "yolov8n.pt"

def load_model(num_classes=1):
    """Load model and set number of classes to num_classes (default model has 80 classes)"""
    model = YOLO(MODEL_PATH)
    if model.model.nc != num_classes:
        model.model.nc = num_classes
        from ultralytics.nn.tasks import DetectionModel
        model.model = DetectionModel(model.model.yaml, nc=num_classes).to(model.device)
    return model

def _get_trainable_keys(model):
    """Normal FL: Returns keys for all parameters that require gradient."""
    return [k for k, v in model.model.state_dict().items()]

def get_parameters(model):
    """Normal FL: share ALL model tensors."""
    state = model.model.state_dict()
    keys = _get_trainable_keys(model)
    return [state[k].detach().cpu().numpy().copy() for k in keys]

def set_parameters(model, parameters):
    """Normal FL: Update the entire model with global weights."""
    current_state = model.model.state_dict()
    keys = _get_trainable_keys(model)

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

