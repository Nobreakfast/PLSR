import torch.nn as nn
import torch.nn.init as init
from .core import resnet2 as models
import torch.nn.utils.prune as prune


# TODO the pretrained is not used for the customized model,
# and this is used for the name of layer. Optimize this part later
def resnet8(pretrained, num_classes):
    model = models.resnet20()
    model.fc = nn.Linear(64, num_classes)
    if pretrained != "no":
        module = get_module(pretrained, model)
        prune.random_unstructured(module, name="weight", amount=0.995)
    return model


def get_module(name, model):
    name_list = name.split(".")
    temp = getattr(model, name_list[0])
    for i in range(len(name_list) - 1):
        temp = getattr(temp, name_list[i + 1])
    return temp
