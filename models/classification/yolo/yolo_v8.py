from ultralytics import YOLO
import torch.nn as nn

"""The Yolo mod"""

class YOLO_v8n(nn.Module):
    def __init__(self, input_channel_num, output_channel_num, use_pretrained=False, dropout=0.0):
        super(YOLO_v8n, self).__init__()
        self.model = YOLO("models/pretrained/yolov8n-cls.pt").model if use_pretrained else YOLO("yolov8n-cls.yaml").model
        self.model.model[0].conv = nn.Conv2d(input_channel_num, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.model[9].drop = nn.Dropout(p=dropout, inplace=True)
        if output_channel_num == 1:
            raise Warning("Not recommended to set the output_channel_num to 1.")
        self.model.model[9].linear = nn.Linear(in_features=1280, out_features=output_channel_num, bias=True)

    def forward(self, x):
        logit = self.model(x)
        return logit
    
class YOLO_v8s(nn.Module):
    def __init__(self, input_channel_num, output_channel_num, use_pretrained=False, dropout=0.0):
        super(YOLO_v8s, self).__init__()
        self.model = YOLO("models/pretrained/yolov8s-cls.pt").model if use_pretrained else YOLO("yolov8s-cls.yaml").model
        self.model.model[0].conv = nn.Conv2d(input_channel_num, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.model[9].drop = nn.Dropout(p=dropout, inplace=True)
        if output_channel_num == 1:
            raise Warning("Not recommended to set the output_channel_num to 1.")
        self.model.model[9].linear = nn.Linear(in_features=1280, out_features=output_channel_num, bias=True)

    def forward(self, x):
        logit = self.model(x)
        return logit
    
class YOLO_v8m(nn.Module):
    def __init__(self, input_channel_num, output_channel_num, use_pretrained=False, dropout=0.0):
        super(YOLO_v8m, self).__init__()
        self.model = YOLO("models/pretrained/yolov8m-cls.pt").model if use_pretrained else YOLO("yolov8m-cls.yaml").model
        self.model.model[0].conv = nn.Conv2d(input_channel_num, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.model[9].drop = nn.Dropout(p=dropout, inplace=True)
        if output_channel_num == 1:
            raise Warning("Not recommended to set the output_channel_num to 1.")
        self.model.model[9].linear = nn.Linear(in_features=1280, out_features=output_channel_num, bias=True)

    def forward(self, x):
        logit = self.model(x)
        return logit
    
class YOLO_v8l(nn.Module):
    def __init__(self, input_channel_num, output_channel_num, use_pretrained=False, dropout=0.0):
        super(YOLO_v8l, self).__init__()
        self.model = YOLO("models/pretrained/yolov8l-cls.pt").model if use_pretrained else YOLO("yolov8l-cls.yaml").model
        self.model.model[0].conv = nn.Conv2d(input_channel_num, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.model[9].drop = nn.Dropout(p=dropout, inplace=True)
        if output_channel_num == 1:
            raise Warning("Not recommended to set the output_channel_num to 1.")
        self.model.model[9].linear = nn.Linear(in_features=1280, out_features=output_channel_num, bias=True)

    def forward(self, x):
        logit = self.model(x)
        return logit
    
class YOLO_v8x(nn.Module):
    def __init__(self, input_channel_num, output_channel_num, use_pretrained=False, dropout=0.0):
        super(YOLO_v8x, self).__init__()
        self.model = YOLO("models/pretrained/yolov8x-cls.pt").model if use_pretrained else YOLO("yolov8x-cls.yaml").model
        self.model.model[0].conv = nn.Conv2d(input_channel_num, 80, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.model[9].drop = nn.Dropout(p=dropout, inplace=True)
        if output_channel_num == 1:
            raise Warning("Not recommended to set the output_channel_num to 1.")
        self.model.model[9].linear = nn.Linear(in_features=1280, out_features=output_channel_num, bias=True)

    def forward(self, x):
        logit = self.model(x)
        return logit