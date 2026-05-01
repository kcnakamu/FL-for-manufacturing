import torch
from ultralytics import YOLO
from collections import OrderedDict

MODEL_PATH = "yolov8n.pt"


def load_model(num_classes=1):
    model = YOLO(MODEL_PATH)
    model.model.yaml['nc'] = num_classes
    model.overrides["model"] = MODEL_PATH
    return model

def get_parameters(model):
    return [val.cpu().numpy().copy() for _, val in model.model.state_dict().items()]

def set_parameters(model, parameters):
    current_state = model.model.state_dict()
    keys = list(current_state.keys())
    if len(parameters) != len(keys):
        raise ValueError(
            f"Parameter count mismatch: got {len(parameters)} arrays, expected {len(keys)}. "
            "Check that server and clients use the same model/num_classes."
        )

    updated_state = OrderedDict()
    for k, v in zip(keys, parameters):
        src = torch.from_numpy(v.copy())
        ref = current_state[k]
        updated_state[k] = src.to(device=ref.device, dtype=ref.dtype)

    model.model.load_state_dict(updated_state, strict=True)