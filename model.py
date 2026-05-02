import torch
from ultralytics import YOLO
from collections import OrderedDict

MODEL_PATH = "yolov8n.pt"
BACKBONE_MAX_LAYER_IDX = 9


def load_model(num_classes=1):
    model = YOLO(MODEL_PATH)
    # model.model.yaml['nc'] = num_classes
    # model.overrides["model"] = MODEL_PATH
    # return model
    if model.model.nc != num_classes:
        model.model.nc = num_classes
        # This re-generates the detection layers (Layer 22) with correct shapes
        from ultralytics.nn.tasks import DetectionModel
        # Re-build the model architecture with the new nc
        model.model = DetectionModel(model.model.yaml, nc=num_classes).to(model.device)
    return model

def _backbone_state_keys(state_dict):
    keys = []
    for k in state_dict.keys():
        parts = k.split(".")

        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            layer_idx = int(parts[1])
            if layer_idx <= BACKBONE_MAX_LAYER_IDX:
                if any(x in k for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                    continue 

                keys.append(k)
    print(f"backbone keys: {keys}")
    return keys

def get_parameters(model):
    """Personalized FL: share only backbone tensors."""
    state = model.model.state_dict()
    keys = _backbone_state_keys(state)
    return [state[k].detach().cpu().numpy().copy() for k in keys]

def set_parameters(model, parameters):
    current_state = model.model.state_dict()
    keys = _backbone_state_keys(current_state)

    print(f"expected keys: {keys}")

    print(f"\n=== DEBUG set_parameters ===")
    print(f"Received {len(parameters)} parameter arrays from server")
    print(f"Expected {len(keys)} backbone keys locally")
    print(f"Local backbone keys: {keys[:5]}... (showing first 5)")
    print(f"Model path: {MODEL_PATH}, Backbone max layer: {BACKBONE_MAX_LAYER_IDX}")
    print("=" * 40 + "\n")

    if len(parameters) != len(keys):
        raise ValueError(
            f"Parameter count mismatch: got {len(parameters)} arrays, expected {len(keys)}. "
            "Check that server and clients use the same personalized-backbone setup."
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

    # Partial load: only backbone tensors are synchronized globally.
    model.model.load_state_dict(updated_state, strict=False)