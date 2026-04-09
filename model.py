import torch
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"

def load_model():
    return YOLO(MODEL_PATH)

def get_parameters(model):
    return [p.data.cpu().numpy().copy() for p in model.model.parameters()]

def set_parameters(model, parameters):
    for param, value in zip(model.model.parameters(), parameters):
        param.data = torch.from_numpy(value.copy()).to(param.device)