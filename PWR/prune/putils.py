import torch
import torch.nn.utils.prune as prune
from hydra.utils import to_absolute_path as abspath
import logging
from PWR import utils

log = logging.getLogger("Putils")


def get_mask_dict(prune_dict: dict) -> dict:
    mask_dict = {}
    for key, module in prune_dict.items():
        mask_dict[key] = module.weight_mask.data.clone().detach()
    return mask_dict


def remove_mask(prune_dict: dict) -> None:
    for key, module in prune_dict.items():
        prune.remove(module, "weight")


def change_mask(prune_dict: dict, mask_dict: dict) -> None:
    for key, module in prune_dict.items():
        module.weight_mask.data *= mask_dict[key]
        module.weight.data = module.weight_orig.data * module.weight_mask.data.float()


def apply_mask(prune_dict: dict, mask_dict: dict) -> None:
    for key, module in prune_dict.items():
        Foo.apply(module, "weight")
        module.weight_mask.data *= mask_dict[key]
        module.weight.data = module.weight_orig.data * module.weight_mask.data.float()


def apply_fooweight(prune_dict: dict) -> None:
    for key, module in prune_dict.items():
        FooWeight.apply(module, "weight")


def apply_foo(prune_dict: dict) -> None:
    for key, module in prune_dict.items():
        Foo.apply(module, "weight")


class Unstructured(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, indices):
        self.indices = indices

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if self.indices != None:
            mask.view(-1)[self.indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, indices):
        return super(Unstructured, cls).apply(module, name, indices=indices)


class Foo(prune.BasePruningMethod):
    """
    add Null mask into module, which means the attribute .weight_mask are all 1
    """

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        return default_mask

    @classmethod
    def apply(cls, module, name):
        return super(Foo, cls).apply(module, name)


class FooWeight(prune.BasePruningMethod):
    """
    add mask based on the zero's in weight
    """

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        mask *= t.data != 0
        return mask

    @classmethod
    def apply(cls, module, name):
        return super(FooWeight, cls).apply(module, name)


def check_sparsity(prune_dict: dict, iter) -> float:
    """
    print the overall sparsity of the model, use log to print
    """
    total_pruned = 0
    total_element = 0
    for key, module in prune_dict.items():
        total_pruned += torch.sum(module.weight_mask.data == 0)
        total_element += module.weight_mask.data.numel()
    total_sparsity = total_pruned / total_element
    log.info(
        f"Iter: {iter:2d}, Original: {total_element/1e6:2.2f}M, Pruned: {total_pruned/1e6:2.2f}M, Sparsity: {total_sparsity:2.2%}"
    )
    return total_sparsity


def check_sparsity_byweight(prune_dict: dict, iter) -> float:
    """
    print the overall sparsity of the model, use log to print
    """
    total_pruned = 0
    total_element = 0
    for key, module in prune_dict.items():
        total_pruned += torch.sum(module.weight.data == 0)
        total_element += module.weight.data.numel()
    total_sparsity = total_pruned / total_element
    log.info(
        f"Iter: {iter:2d}, Original: {total_element/1e6:2.2f}M, Pruned: {total_pruned/1e6:2.2f}M, Sparsity: {total_sparsity:2.2%}"
    )
    return total_sparsity


def check_layer_sparsity(prune_dict: dict) -> dict:
    """
    Check the sparsity of each layer
    """
    total_pruned = 0
    total_element = 0
    sparsity_dict = {}
    for key, module in prune_dict.items():
        pruned = torch.sum(module.weight_mask.data == 0)
        element = module.weight_mask.data.numel()
        sparsity = pruned / element
        sparsity_dict[key] = sparsity
        log.info(
            f"{key}: Original: {element/1e6:2.2f}M, Pruned: {pruned/1e6:2.2f}M, Sparsity: {sparsity:2.2%}"
        )
        total_pruned += pruned
        total_element += element
    total_sparsity = total_pruned / total_element
    log.info(
        f"Original: {total_element / 1e6:2.2f}M, Pruned: {total_pruned / 1e6:2.2f}M, Sparsity: {total_sparsity:2.2%}"
    )
    return sparsity_dict


def shuffle_mask(prune_dict) -> None:
    """
    shuffle the mask of each layer
    """
    num_toprune = {}
    for name, module in prune_dict.items():
        amount = int(torch.sum(module.weight_mask.data == 0))
        num_toprune[name] = amount
        module.weight_mask.data = torch.ones_like(module.weight_mask.data)
        module.weight = module.weight_orig * module.weight_mask
    remove_mask(prune_dict)
    for key, module in prune_dict.items():
        prune.random_unstructured(module, "weight", num_toprune[key])
    log.info("shuffled the mask")


def reinit_weight(prune_dict, model) -> None:
    """
    reinit the weight but save the mask
    """
    mask_dict = get_mask_dict(prune_dict)
    remove_mask(prune_dict)
    utils.reset_parameters(model)
    apply_mask(prune_dict, mask_dict)
    log.info("reinitialized the weight")


def init_weight(prune_dict, train_handler) -> None:
    """
    load init weight and save the mask
    """
    mask_dict = get_mask_dict(prune_dict)
    remove_mask(prune_dict)
    train_handler.load_checkpoint(
        #"init", abspath(train_handler.cfg["model"]["load_path"])
        "init", "./saved"
    )
    apply_mask(prune_dict, mask_dict)
    log.info("initialized the weight")


def restore(prune_dict, train_handler, name) -> None:
    """
    restore the weight distribution
    """
    remove_mask(prune_dict)
    present_state = train_handler.get_current_state([True, False, False])
    train_handler.load_checkpoint(name, train_handler.cfg["model"]["load_path"])
    mean_dict = {}
    var_dict = {}
    for key, module in prune_dict.items():
        mean_dict[key] = module.weight.data.mean()
        var_dict[key] = module.weight.data.var()
    train_handler.load_state(present_state, [True, False, False])
    for key, module in prune_dict.items():
        rho = 1 - torch.sum(module.weight.data == 0) / module.weight.data.numel()
        FooWeight.apply(module, "weight")
        if rho == 0:
            continue
        var_ratio = var_dict[key] / module.weight.data.var()
        module.weight_orig.data *= var_ratio**0.5
        module.weight.data = module.weight_orig.data * module.weight_mask.data.float()
        mean_cut = module.weight.data.mean() / rho
        module.weight_orig.data -= mean_cut * module.weight_mask.data.float()
        module.weight.data = module.weight_orig.data * module.weight_mask.data.float()
    log.info(f"restored the weight to {name}.pth")


def prune_init(prune_dict, train_handler) -> None:
    """
    reinit the weight for none zero weight
    and the none zero weight have same mean and var as the original weight
    """
    for key, module in prune_dict.items():
        weight_nonzero = module.weight.data != 0
        num_saved = int(torch.sum(weight_nonzero))
        num_all = module.weight.data.numel()
        std = module.weight_orig.data.std()# * ((num_saved / num_all) ** 0.5)
        module.weight_orig.data[weight_nonzero] = (
            torch.rand(num_saved).to(train_handler.basic_dict["device"]) * std
        )
        module.weight.data = module.weight_orig.data * module.weight_mask.data.float()

    # mask_dict = get_mask_dict(prune_dict)
    # remove_mask(prune_dict)

    # apply_mask(prune_dict, mask_dict)
    # log.info("initialized the weight")
