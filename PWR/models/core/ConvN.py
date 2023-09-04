import torch
import torch.nn as nn
import torch.nn.init as init


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bn: bool,
        bias=False,
    ):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if hasattr(self, "bn") else x
        return self.relu(x)


class CONVN(nn.Module):
    def __init__(self, input: tuple, layer: int, bn: bool, num_classes: int):
        super(CONVN, self).__init__()
        feature_list = []
        feature_list.append(BasicBlock(3, 2**4, 3, 1, 1, bn=bn, bias=False))
        for i in range(layer - 1):
            feature_list.append(
                BasicBlock(2 ** (i + 4), 2 ** (i + 5), 3, 1, 1, bn=bn, bias=False)
            )
        feature_list.append(nn.Flatten())
        self.features = nn.Sequential(*feature_list)

        classifier_list = []
        classifier_list.append(nn.Linear(2 ** (layer + 3) * input[0] * input[1], 512))
        classifier_list.append(nn.ReLU(inplace=True))
        classifier_list.append(nn.Linear(512, num_classes))
        self.classifier = nn.Sequential(*classifier_list)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.xavier_normal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
