import torch.nn as nn
import torch.nn.init as init
from .core import resnet2 as models


def resnet20(pretrained, num_classes):
    model = models.resnet20()
    model.fc = nn.Linear(64, num_classes)
    return model
