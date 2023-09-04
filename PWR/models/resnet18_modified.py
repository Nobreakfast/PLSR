import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


def resnet18_modified(pretrained, num_classes):
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.fc = nn.Linear(512, num_classes)
    return model
