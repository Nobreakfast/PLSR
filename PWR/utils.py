import torch
import torch.nn as nn

import PWR


def get_obj(obj: str):
    """
    Returns the object from the string.
    """
    obj_list = obj.split(".")
    if len(obj_list) < 2:
        raise ValueError("The object name is invalid.")
    source = __import__(obj_list[0])
    for i in range(len(obj_list) - 1):
        temp = getattr(source, obj_list[i + 1])
        source = temp
    return source


def get_basic_dict(cfg) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = get_obj(cfg.model.func)(**cfg.model.params)
    reset_parameters(model)
    model.to(device)
    train_loader, test_loader = get_obj(cfg.data.func)(**cfg.data.params)
    if cfg.criterion.params is not None:
        criterion = get_obj(cfg.criterion.func)(**cfg.criterion.params)
    else:
        criterion = get_obj(cfg.criterion.func)()
    optimizer = get_obj(cfg.optimizer.func)(model.parameters(), **cfg.optimizer.params)
    if type(cfg.scheduler) != str:
        scheduler = get_obj(cfg.scheduler.func)(optimizer, **cfg.scheduler.params)
    else:
        scheduler = None
    return {
        "device": device,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


# def get_prune_dict(cfg_prune: dict) -> dict:
#     method = get_obj(cfg_prune["method"]["name"])
#     score = get_obj(cfg_prune["score"]["name"])
#     return {
#         "method": method,
#         "score": score,
#     }


def get_module_dict(model: nn.Module) -> dict:
    module_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module_dict[name] = module
    return module_dict


def get_cfg_data(cfg) -> dict:
    for key in cfg.keys():
        if hasattr(cfg[key], "keys"):
            cfg[key] = get_cfg_data(cfg[key])
        cfg[key] = cfg[key]
    return cfg


def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.xavier_normal_(m.weight, gain=1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
