from torchvision.models import resnet101
import torch.nn as nn

class ResNet101(nn.Module):
    def __init__(self, input_channels, output_channel_num, use_pretrained=False, dropout=0):
        super(ResNet101, self).__init__()
        self.model = resnet101(weights = use_pretrained)
        # Insure input channel

        # Insure output channel

        # Make dropout rate adjustable

    def forward(self, x):
        logit = self.model(x)
        return logit