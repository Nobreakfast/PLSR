import os

import hydra
from tqdm.contrib.logging import logging_redirect_tqdm
from hydra.utils import to_absolute_path

import scripts


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    with logging_redirect_tqdm():
        script = getattr(scripts, cfg.script)
        script.main(cfg)
        with open(f"{to_absolute_path('logs')}/{cfg.date}.txt", "a") as f:
            f.write(f"{cfg.name}: {os.getcwd()}\n")

if __name__ == "__main__":
    main()
