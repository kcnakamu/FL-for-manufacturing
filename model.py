import torch
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"

def load_model(num_classes=6):
    model = YOLO(MODEL_PATH)
    # force the model to use the correct number of classes
    model.model.yaml['nc'] = num_classes
    return model

def get_parameters(model):
    return [p.data.cpu().numpy().copy() for p in model.model.parameters()]

def set_parameters(model, parameters):
    for param, value in zip(model.model.parameters(), parameters):
        param.data = torch.from_numpy(value.copy()).to(param.device)