import torch
from ultralytics import YOLO
from collections import OrderedDict

MODEL_PATH = "yolov8n.pt"
strategy = "BN"

def load_model(num_classes=2):
    model = YOLO(MODEL_PATH)
    model.model.yaml['nc'] = num_classes
    return model

def get_parameters(model):
    if strategy == "BN":
        return [
            val.cpu().numpy().copy() 
            for name, val in model.model.state_dict().items() 
            if "bn" not in name.lower()
        ] # BN
    else:
        return [p.data.cpu().numpy().copy() for p in model.model.parameters()] # Regular FL

def set_parameters(model, parameters):
    model.model.train()
    if strategy == "BN":
        params_dict = zip(
            [name for name in model.model.state_dict().keys() if "bn" not in name.lower()],
            parameters
        )
        device = next(model.model.parameters()).device
        
        state_dict = OrderedDict({
            k: torch.from_numpy(v.copy()).to(device) 
            for k, v in params_dict
        })
        
        model.model.load_state_dict(state_dict, strict=False)
    else:
        for param, value in zip(model.model.parameters(), parameters):
            param.data = torch.from_numpy(value.copy()).to(param.device)