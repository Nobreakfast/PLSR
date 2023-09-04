"""
from discussion in slack by @Liye:
https://nobreakfast-workspace.slack.com/archives/D03H116E606/p1655194034206349?thread_ts=1655178344.818489&cid=D03H116E606
"""
import torch
import torch.nn as nn
from PWR import Trainer, Pruner
import matplotlib.pyplot as plt
import logging

log = logging.getLogger("exp")


def print_distribution(module, key):
    # collect the remaining weights
    non_zeros = module.weight.data[module.weight.data != 0]
    log.info(
        f"Module: {key:22}, ori: {torch.mean(module.weight_orig.data):.4f}, {torch.std(module.weight_orig.data):.4f} || "
        f"pruned: {torch.mean(module.weight.data):.4f}, {torch.std(module.weight.data):.4f} || "
        f"rem: {torch.mean(non_zeros):.4f}, {torch.std(non_zeros):.4f}"
    )


def main(cfg):
    train_handler = Trainer.TrainHandler(cfg)
    prune_handler = Pruner.PruneHandler(cfg["prune"], train_handler)
    prune_handler.run()
    train_handler.writer.close()
    for key, module in prune_handler.prune_dict.items():
        print_distribution(module, key)
