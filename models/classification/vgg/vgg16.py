from torchvision.models import vgg16
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, input_channels, output_channel_num, use_pretrained=False, dropout=0):
        super(VGG16, self).__init__()
        '''
        SqueezeNet(
            (features): Sequential(
                (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2))
                (1): ReLU(inplace=True)
                (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
                (3): Fire(
                (squeeze): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
                (squeeze_activation): ReLU(inplace=True)
                (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
                (expand1x1_activation): ReLU(inplace=True)
                (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (expand3x3_activation): ReLU(inplace=True)
                )
                (4): Fire(
                (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
                (squeeze_activation): ReLU(inplace=True)
                (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
                (expand1x1_activation): ReLU(inplace=True)
                (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (expand3x3_activation): ReLU(inplace=True)
                )
                (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
                (6): Fire(
                (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
                (squeeze_activation): ReLU(inplace=True)
                (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
                (expand1x1_activation): ReLU(inplace=True)
                (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (expand3x3_activation): ReLU(inplace=True)
                )
                (7): Fire(
                (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
                (squeeze_activation): ReLU(inplace=True)
                (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
                (expand1x1_activation): ReLU(inplace=True)
                (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (expand3x3_activation): ReLU(inplace=True)
                )
                (8): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
                (9): Fire(
                (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
                (squeeze_activation): ReLU(inplace=True)
                (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
                (expand1x1_activation): ReLU(inplace=True)
                (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (expand3x3_activation): ReLU(inplace=True)
                )
                (10): Fire(
                (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
                (squeeze_activation): ReLU(inplace=True)
                (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
                (expand1x1_activation): ReLU(inplace=True)
                (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (expand3x3_activation): ReLU(inplace=True)
                )
                (11): Fire(
                (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
                (squeeze_activation): ReLU(inplace=True)
                (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
                (expand1x1_activation): ReLU(inplace=True)
                (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (expand3x3_activation): ReLU(inplace=True)
                )
                (12): Fire(
                (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
                (squeeze_activation): ReLU(inplace=True)
                (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
                (expand1x1_activation): ReLU(inplace=True)
                (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (expand3x3_activation): ReLU(inplace=True)
                )
            )
            (classifier): Sequential(
                (0): Dropout(p=0.5, inplace=False)
                (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
                (2): ReLU(inplace=True)
                (3): AdaptiveAvgPool2d(output_size=(1, 1))
            )
        )
        '''
        self.model = vgg16(weights = use_pretrained)
        # Insure input channel
        self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(2, 2))
        # Insure output channel
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=output_channel_num, bias=True)
        # Make dropout rate adjustable
        self.model.classifier[2] = nn.Dropout(p=dropout, inplace=False)
        self.model.classifier[5] = nn.Dropout(p=dropout, inplace=False)
        # # Add dropout
        # feats_list = list(self.model.features)
        # new_feats_list = []
        # for feat in feats_list:
        #     new_feats_list.append(feat)
        #     if isinstance(feat, nn.Conv2d):
        #         new_feats_list.append(nn.Dropout(p = dropout, inplace = True))

    def forward(self, x):
        logit = self.model(x)
        return logit