# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class VarMobileNetV1(nn.Module):
    def __init__(self, num_classes=1024, input_channels=3, start_fm=32):
        super(VarMobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(input_channels, start_fm, 2),
            conv_dw(start_fm, start_fm*2, 1),
            conv_dw(start_fm*2**1, start_fm*2**2, 2),
            conv_dw(start_fm*2**2, start_fm*2**2, 1),
            conv_dw(start_fm*2**2, start_fm*2**3, 2),
            conv_dw(start_fm*2**3, start_fm*2**3, 1),
            conv_dw(start_fm*2**3, start_fm*2**4, 2),
            conv_dw(start_fm*2**4, start_fm*2**4, 1),
            conv_dw(start_fm*2**4, start_fm*2**4, 1),
            conv_dw(start_fm*2**4, start_fm*2**4, 1),
            conv_dw(start_fm*2**4, start_fm*2**4, 1),
            conv_dw(start_fm*2**4, start_fm*2**4, 1),
            conv_dw(start_fm*2**4, start_fm*2**5, 2),
            conv_dw(start_fm*2**5, start_fm*2**5, 1),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x