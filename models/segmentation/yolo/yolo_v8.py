from ultralytics import YOLO
import torch.nn as nn

"""The Yolo mod"""

class YOLO_v8(nn.Module):
    def __init__(self, use_pretrained = False):
        super(YOLO_v8, self).__init__()
        self.model = YOLO("models/pretrained/yolov8n-seg.pt").model if use_pretrained else YOLO("models/pretrained/yolov8n-seg.yaml").model

    def forward(self, x):
        logit = self.model(x)
        return logit