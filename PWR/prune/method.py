import torch
import logging
from tqdm import tqdm
import sys
from . import putils

log = logging.getLogger("Prune Method")
prune_count = 0


def prune_indices(prune_dict, score_dict, amount, inverse=False) -> dict:
    indices_dict = _get_indices_dict(prune_dict.keys(), score_dict, amount, inverse)
    _apply_indices(prune_dict, indices_dict)
    return indices_dict


def prune_once(prune_dict, score_fn, cfg_prune, amount, train_handler):
    putils.apply_fooweight(prune_dict)
    putils.remove_mask(prune_dict)
    prune_indices(
        prune_dict,
        score_fn(
            prune_dict,
            cfg_prune["score"],
            train_handler,
        ),
        amount,
        cfg_prune["inverse"],
    )


def oneshot(prune_handler):
    """
    One-shot pruning
    """
    log.info("One-shot pruning Started")
    prune_once(
        prune_handler.prune_dict,
        prune_handler.score_fn,
        prune_handler.cfg_prune,
        prune_handler.cfg_prune["amount"],
        prune_handler.train_handler,
    )
    prune_handler.check_sparsity()
    if prune_handler.cfg_prune["plot_prune_stat"]:
        prune_handler.plot_prune_stat()
    prune_handler.prune_count += 1
    log.info("One-shot pruning Finished")


def lth(prune_handler):
    """
    LTH pruning pipeline
    """
    log.info("LTH pruning Started")
    init_state = prune_handler.train_handler.get_current_state([True, True, True])
    amount_temp = prune_handler.cfg_prune["amount"] * (
        1 / prune_handler.cfg_prune["method"]["iterations"]
    )
    epoch = prune_handler.cfg_prune["method"]["epoch"]
    for i in range(prune_handler.cfg_prune["method"]["iterations"]):
        prune_handler.train_for(epoch)
        amount = amount_temp * (i + 1)
        prune_once(
            prune_handler.prune_dict,
            prune_handler.score_fn,
            prune_handler.cfg_prune,
            amount,
            prune_handler.train_handler,
        )
        prune_handler.check_sparsity()
        if prune_handler.cfg_prune["plot_prune_stat"]:
            prune_handler.plot_prune_stat()
        prune_handler.prune_count += 1
        mask_dict = putils.get_mask_dict(prune_handler.prune_dict)
        putils.remove_mask(prune_handler.prune_dict)
        prune_handler.train_handler.load_state(init_state, [True, True, True])
        putils.apply_mask(prune_handler.prune_dict, mask_dict)
    log.info("LTH pruning Finished")


def iterative(prune_handler):
    """
    Iteration pruning
    """
    log.info("Iterative pruning Started")
    #init_state = prune_handler.train_handler.get_current_state([False, True, True])
    amount_temp = prune_handler.cfg_prune["amount"] * (
        1 / prune_handler.cfg_prune["method"]["iterations"]
    )
    epoch = prune_handler.cfg_prune["method"]["epoch"]
    for i in range(prune_handler.cfg_prune["method"]["iterations"]):
        prune_handler.train_for(epoch)
        amount = amount_temp * (i + 1)
        prune_once(
            prune_handler.prune_dict,
            prune_handler.score_fn,
            prune_handler.cfg_prune,
            amount,
            prune_handler.train_handler,
        )
        prune_handler.check_sparsity()
        if prune_handler.cfg_prune["plot_prune_stat"]:
            prune_handler.plot_prune_stat()
        prune_handler.prune_count += 1
        #prune_handler.train_handler.load_state(init_state, [False, True, True])
    log.info("Iterative pruning Finished")


def pai(prune_handler):
    """
    Pruning at Initialization
    """
    log.info("Pruning at Initialization Started")
    amount_temp = prune_handler.cfg_prune["amount"] * (
        1 / prune_handler.cfg_prune["method"]["iterations"]
    )
    for i in range(prune_handler.cfg_prune["method"]["iterations"]):
        amount = amount_temp * (i + 1)
        prune_once(
            prune_handler.prune_dict,
            prune_handler.score_fn,
            prune_handler.cfg_prune,
            amount,
            prune_handler.train_handler,
        )
        prune_handler.check_sparsity()
        if prune_handler.cfg_prune["plot_prune_stat"]:
            prune_handler.plot_prune_stat()
        prune_handler.prune_count += 1
    log.info("Pruning at Initialization Finished")


def pai_with_ratio(prune_handler, num_toprune, score_fn):
    """
    Pruning at Initialization with Ratio
    """
    log.info("Pruning at Initialization with Ratio Started")

    for i in range(prune_handler.cfg_prune["pai_new_iterations"]):
        tmp_num_toprune = {}
        for key in num_toprune.keys():
            tmp_num_toprune[key] = int(
                num_toprune[key]
                * (i + 1)
                / prune_handler.cfg_prune["pai_new_iterations"]
            )
        # print(tmp_num_toprune)
        # print(num_toprune)
        putils.apply_fooweight(prune_handler.prune_dict)
        putils.remove_mask(prune_handler.prune_dict)
        score_dict = score_fn(
            prune_handler.prune_dict,
            prune_handler.cfg_prune["score"],
            prune_handler.train_handler,
        )
        indices_dict = _get_indices_dict_with_ratio(
            prune_handler.prune_dict.keys(),
            score_dict,
            tmp_num_toprune,
            prune_handler.cfg_prune["inverse"],
        )
        _apply_indices(prune_handler.prune_dict, indices_dict)
        prune_handler.check_sparsity()
        if prune_handler.cfg_prune["plot_prune_stat"]:
            prune_handler.plot_prune_stat()
        prune_handler.prune_count += 1
    log.info("Pruning at Initialization Finished")


def pwr(prune_dict, num_toprune, score_fn, cfg_prune):
    """
    Pruning with Ratio
    """
    log.info("Pruning with Ratio L1 Started")
    # prune_dict = {key: module}
    # prune_once(prune_dict, score_fn, cfg_prune, num_toprune, None)

    putils.apply_fooweight(prune_dict)
    putils.remove_mask(prune_dict)
    score_dict = score_fn(
        prune_dict,
        cfg_prune["score"],
        None,
    )
    indices_dict = _get_indices_dict_with_ratio(
        prune_dict.keys(),
        score_dict,
        num_toprune,
        cfg_prune["inverse"],
    )
    _apply_indices(prune_dict, indices_dict)
    log.info("Pruning with Ratio Finished")


def erk(prune_handler):
    """
    Pruning with Ratio
    """
    log.info("Pruning with Ratio Started")
    amount = prune_handler.cfg_prune["amount"]
    sparsity_dict = get_sparsity(
        prune_handler.prune_dict,
        prune_handler.cfg_prune["amount"],
    )
    for key, module in prune_handler.prune_dict.items():
        prune_dict = {key: module}
        prune_once(
            prune_dict,
            prune_handler.score_fn,
            prune_handler.cfg_prune,
            amount * sparsity_dict[key],
            prune_handler.train_handler,
        )
    prune_handler.check_sparsity()
    if prune_handler.cfg_prune["plot_prune_stat"]:
        prune_handler.plot_prune_stat()
    prune_handler.prune_count += 1
    log.info("Pruning with Ratio Finished")


def get_sparsity(module_dict, amount):
    """
    Calcualte the sparsity of the module
    """
    sparsity_dict = {}
    for key, module in module_dict.items():
        if isinstance(module, torch.nn.Conv2d):
            in_ch = module.in_channels
            out_ch = module.out_channels
            h, w = module.kernel_size
            h_add_w = h + w
            h_mul_w = h * w
        else:
            in_ch = module.in_features
            out_ch = module.out_features
            h_add_w = 0
            h_mul_w = 1

        sparsity_dict[key] = 1 - (in_ch + h_add_w + out_ch) / (in_ch * out_ch * h_mul_w)
    return sparsity_dict

#def _get_indices_dict(keys, score_dict, amount, inverse):
#    score_list = torch.cat(tuple([score_dict[key] for key in score_dict.keys()]), dim=0)
#    if not inverse:
#        number = int(amount * score_list.size(0)) if type(amount) == float else amount
#    else:
#        number = (
#            int((1 - amount) * score_list.size(0))
#            if type(amount) == float
#            else score_list.size(0) - amount
#        )
#    print(number)
#    if number == 0:
#        return {key: None for key in keys}
#    kth = torch.kthvalue(score_list, number)[0]
#
#    indices_dict = {}
#    for key in keys:
#        indices_dict[key] = torch.nonzero(
#            torch.le(score_dict[key], kth)
#            if not inverse
#            else torch.ge(score_dict[key], kth)
#        ).squeeze()
#    return indices_dict


def _get_indices_dict(keys, score_dict, amount, inverse):
    if inverse:
        for key in keys:
          score_dict[key][score_dict[key]==0] = 99
          score_dict[key] = -score_dict[key]
          #print(score_dict[key][:10])
    score_list = torch.cat(tuple([score_dict[key] for key in score_dict.keys()]), dim=0)
    number = int(amount * score_list.size(0)) if type(amount) == float else amount
    #print(number)
    if number == 0:
        return {key: None for key in keys}
    kth = torch.kthvalue(score_list, number)[0]
    #print(kth)

    indices_dict = {}
    for key in keys:
        indices_dict[key] = torch.nonzero(
            torch.le(score_dict[key], kth)
        ).squeeze()
    return indices_dict


def _get_indices_dict_with_ratio(keys, score_dict, num_toprune, inverse):
    indices_dict = {}
    for key in keys:
        score_list = score_dict[key]
        # print(score_list.size(0), num_toprune[key])
        amount = num_toprune[key]
        if not inverse:
            number = (
                int(amount * score_list.size(0)) if type(amount) == float else amount
            )
        else:
            number = (
                int((1 - amount) * score_list.size(0))
                if type(amount) == float
                else score_list.size(0) - amount
            )
        if number == 0:
            indices_dict[key] = None
            continue
        kth = torch.kthvalue(score_list, number)[0]

        indices_dict[key] = torch.nonzero(
            torch.le(score_dict[key], kth)
            if not inverse
            else torch.ge(score_dict[key], kth)
        ).squeeze()
    return indices_dict


def _apply_indices(module_dict, indices_dict):
    for key in module_dict.keys():
        putils.Unstructured.apply(module_dict[key], "weight", indices_dict[key])
