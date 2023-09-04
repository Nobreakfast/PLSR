import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


def vgg16(pretrained, num_classes):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
