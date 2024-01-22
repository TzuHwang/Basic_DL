from . import vgg

class Cls_models:
    def __init__(self, model_name, input_channel_num, output_channel_num, use_pretrained, dropout):
        if model_name in vgg.__all__:
            self.model = vgg.__dict__[model_name](input_channel_num, output_channel_num, use_pretrained, dropout)
    def get_model(self):
        return self.model