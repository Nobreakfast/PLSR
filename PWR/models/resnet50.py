import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


def resnet50(pretrained, num_classes):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(2048, num_classes)
    return model
