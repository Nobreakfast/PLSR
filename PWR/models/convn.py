import torch
import torch.nn as nn
import torch.nn.init as init
from .core import ConvN


def convn(**kwargs):
    model = ConvN.CONVN(**kwargs)
    model.reset_parameters()
    return model
