import torch

from PWR import utils
import logging
from hydra.utils import to_absolute_path as abspath

from PWR.prune import putils
from PWR.Trainer import TrainHandler

log = logging.getLogger("Pruner")


class PruneHandler:
    def __init__(self, cfg_prune: dict, train_handler: TrainHandler) -> None:
        self.train_handler = train_handler
        self.cfg_prune = cfg_prune
        self.prune_dict = train_handler.module_dict
        self.method_fn = utils.get_obj(self.cfg_prune["method"]["func"])
        self.score_fn = utils.get_obj(self.cfg_prune["score"]["func"])
        self.prune_count = 0
        self.writer = train_handler.writer
        self.model = train_handler.basic_dict["model"]
        log.info(f"Pruner Initialized")

    def run(self) -> None:
        if self.cfg_prune["load_model"]:
            self.load_model()
        self.warmup()
        self.prune()
        putils.check_layer_sparsity(self.prune_dict)
        if self.cfg_prune["shuffle"]:
            self.shuffle_mask()
        elif self.cfg_prune["pwr"]:
            self.pwr()
        elif self.cfg_prune["pai_ratio"]:
            self.pai_ratio()
        if self.cfg_prune["init"]:
            self.init_weight()
        elif self.cfg_prune["reinit"]:
            self.reinit_weight()
        elif self.cfg_prune["prune_init"]:
            self.prune_init()
        if self.cfg_prune["restore"]:
            self.restore()
        putils.check_layer_sparsity(self.prune_dict)
        self.finetune()
        putils.check_layer_sparsity(self.prune_dict)
        if not self.cfg_prune["save_mask"]:
            putils.remove_mask(self.prune_dict)

    def prune(self) -> None:
        log.info(f"Pruning Started")
        self.method_fn(self)
        log.info(f"Pruning Finished")

    def load_model(self) -> None:
        log.info(f"Loading Model")
        self.train_handler.load_checkpoint(
            self.cfg_prune["load_name"], self.cfg_prune["load_path"]
        )
        log.info(f"Loading Model Finished")

    def load_init_model(self) -> None:
        log.info(f"Loading Model")
        self.train_handler.load_checkpoint("init", self.cfg_prune["load_path"])
        log.info(f"Loading Model Finished")

    def warmup(self) -> None:
        if self.cfg_prune["method"]["warmup"] != 0:
            init_state = self.train_handler.get_current_state([False, True, True])
            self.train_for(self.cfg_prune["method"]["warmup"])
            self.train_handler.load_state(init_state, [False, True, True])

    def finetune(self) -> None:
        if self.cfg_prune["method"]["finetune"] != 0:
            log.info(f"Finetuning Started")
            self.train_for(self.cfg_prune["method"]["finetune"])
            log.info(f"Finetuning Finished")

    def shuffle_mask(self) -> None:
        putils.shuffle_mask(self.prune_dict)

    def init_weight(self) -> None:
        putils.init_weight(self.prune_dict, self.train_handler)

    def reinit_weight(self) -> None:
        putils.reinit_weight(self.prune_dict, self.model)

    def restore(self) -> None:
        putils.restore(
            self.prune_dict, self.train_handler, self.cfg_prune["restore_name"]
        )

    def prune_init(self) -> None:
        putils.prune_init(self.prune_dict, self.train_handler)

    def pwr(self) -> None:
        num_toprune = {}
        for name, module in self.prune_dict.items():
            amount = int(torch.sum(module.weight_mask.data == 0))
            num_toprune[name] = amount
            module.weight_mask.data = torch.ones_like(module.weight_mask.data)
            module.weight = module.weight_orig * module.weight_mask
        putils.remove_mask(self.prune_dict)
        self.load_model()
        method_fn = utils.get_obj("PWR.prune.method.pwr")
        score_fn = utils.get_obj("PWR.prune.score.l1")
        method_fn(self.prune_dict, num_toprune, score_fn, self.cfg_prune)

    def pai_ratio(self) -> None:
        num_toprune = {}
        for name, module in self.prune_dict.items():
            amount = int(torch.sum(module.weight_mask.data == 0))
            num_toprune[name] = amount
            module.weight_mask.data = torch.ones_like(module.weight_mask.data)
            module.weight = module.weight_orig * module.weight_mask
        putils.remove_mask(self.prune_dict)
        self.load_init_model()
        # putils.check_sparsity_byweight(self.prune_dict, 0)
        method_fn = utils.get_obj("PWR.prune.method.pai_with_ratio")
        score_fn = utils.get_obj("PWR.prune.score." + self.cfg_prune["pai_score"])
        method_fn(self, num_toprune, score_fn)

    def plot_prune_stat(self) -> None:
        """
        plot the prune stat of each layer and plot in tensorboard
            1. weight distribution (remaining)
            2. sparsity
        """
        mean_dict = {}
        var_dict = {}
        sparsity_dict = {}
        for name, module in self.prune_dict.items():
            mean_dict[name] = module.weight.data.mean()
            var_dict[name] = module.weight.data.var()
            pruned = torch.sum(module.weight_mask.data == 0)
            element = module.weight_mask.data.numel()
            sparsity_dict[name] = pruned / element
            self.writer.add_histogram(
                f"prune/weight_distribution/{name}",
                module.weight.data,
                self.prune_count,
            )
        self.writer.add_scalars("prune/mean", mean_dict, self.prune_count)
        self.writer.add_scalars("prune/var", var_dict, self.prune_count)
        self.writer.add_scalars("prune/sparsity", sparsity_dict, self.prune_count)

    def train_for(self, epoch) -> None:
        self.train_handler.train_for(epoch)

    def check_sparsity(self) -> float:
        return putils.check_sparsity(self.prune_dict, self.prune_count)
