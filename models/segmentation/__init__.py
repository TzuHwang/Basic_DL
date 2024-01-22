from . import unet

class Seg_models:
    def __init__(self, model_name, input_channel_num, output_channel_num, bilinear, conv_bn):
        if model_name in unet.__all__:
            self.model = unet.__dict__[model_name](input_channel_num, output_channel_num, bilinear, conv_bn)
    
    def get_model(self):
        return self.model