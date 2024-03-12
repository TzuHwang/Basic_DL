from . import vgg
from . import yolo
from . import graph

class Cls_models:
    def __init__(
            self, model_name, input_channel_num, output_channel_num, use_pretrained, dropout,
            # graph specific args
            hidden_channels=None, num_layers=None
            ):

        if model_name in vgg.__all__:
            self.model = vgg.__dict__[model_name](input_channel_num, output_channel_num, use_pretrained, dropout)
        if model_name in yolo.__all__:
            self.model = yolo.__dict__[model_name](input_channel_num, output_channel_num, use_pretrained, dropout)
        if model_name in graph.__all__:
            self.model = graph.__dict__[model_name](input_channel_num, hidden_channels, output_channel_num, num_layers, dropout)

    def get_model(self):
        return self.model