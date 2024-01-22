from . import vgg, yolo

class Cls_models:
    def __init__(self, model_name, input_channel_num, output_channel_num, use_pretrained, dropout):
        if model_name in vgg.__all__:
            self.model = vgg.__dict__[model_name](input_channel_num, output_channel_num, use_pretrained, dropout)
        elif model_name in yolo.__all__:
            self.model = yolo.__dict__[model_name](use_pretrained)
    def get_model(self):
        return self.model