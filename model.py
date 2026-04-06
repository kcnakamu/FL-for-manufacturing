import torch
from ultralytics import RTDETR

MODEL_PATH = "rtdetr-l.pt"

def load_model():
    return RTDETR(MODEL_PATH)

def get_parameters(model):
    return [p.data.cpu().numpy().copy() for p in model.model.parameters()]

def set_parameters(model, parameters):
    for param, value in zip(model.model.parameters(), parameters):
        param.data = torch.from_numpy(value.copy()).to(param.device)