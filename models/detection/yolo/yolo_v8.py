from ultralytics import YOLO
import torch.nn as nn
import torch

"""The Yolo mod"""

class YOLO_v8(nn.Module):
    def __init__(self, input_channel_num, output_channel_num, use_pretrained = False):
        super(YOLO_v8, self).__init__()
        self.model = YOLO("models/pretrained/yolov8n.pt").model if use_pretrained else YOLO("yolov8n.yaml").model

    def forward(self, x):
        # feat_pyramid, logit = self.model(x)
        logit = self.model(x)
        import pdb;pdb.set_trace()
        return logit