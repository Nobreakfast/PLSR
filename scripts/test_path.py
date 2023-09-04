import os
from hydra.utils import to_absolute_path

def main(cfg):
    print(cfg.prune.load_path)
    os.system(f"ls {cfg.prune.load_path}")
    os.system(f"ls {to_absolute_path(cfg.prune.load_path)}")