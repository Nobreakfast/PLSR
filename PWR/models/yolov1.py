import torch.nn as nn


def yolov1(**kwargs):
    return YoloV1(**kwargs)


class YoloV1(nn.Module):
    def __init__(self, **kwargs):
        super(YoloV1, self).__init__()
        self.num_classes = kwargs.get("class")
        self.num_anchors = kwargs.get("box")
        self.num_split = kwargs.get("split")
