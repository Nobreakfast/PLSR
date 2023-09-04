import os

import torch
import torchvision
import torchvision.transforms as transforms
from .core import custom_datasets


def tinyimagenet(batch_size, path, workers):
    os.system(f"mkdir -p {path}")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.448, 0.397), (0.276, 0.269, 0.282)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.448, 0.397), (0.276, 0.269, 0.282)),
        ]
    )

    trainset = custom_datasets.TINYIMAGENET(
        root=path, train=True, download=True, transform=transform_train
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    testset = custom_datasets.TINYIMAGENET(
        root=path, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size * 8, shuffle=False, num_workers=workers
    )
    return [trainloader, testloader]
