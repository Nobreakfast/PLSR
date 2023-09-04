import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


def resnet18(pretrained, num_classes):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model
