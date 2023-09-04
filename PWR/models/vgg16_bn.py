import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


def vgg16_bn(pretrained, num_classes):
    model = models.vgg16_bn(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
