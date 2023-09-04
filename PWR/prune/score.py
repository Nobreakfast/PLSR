import torch


def l1(module_dict, cfg_score, train_handler=None) -> dict:
    score_dict = {}
    for key in module_dict.keys():
        score_dict.update(
            {key: module_dict[key].weight.data.clone().detach().view(-1).abs()}
        )
    return score_dict


def rand(module_dict, cfg_score, train_handler=None) -> dict:
    import torch

    score_dict = {}
    for key in module_dict.keys():
        score_dict.update(
            {key: torch.rand_like(module_dict[key].weight.data.view(-1)).abs()}
        )
    return score_dict


def randn(module_dict, cfg_score, train_handler=None) -> dict:
    score_dict = {}
    for key in module_dict.keys():
        score_dict.update(
            {key: torch.randn_like(module_dict[key].weight.data.view(-1)).abs()}
        )
    return score_dict


def erk(module_dict, cfg_score, train_handler=None) -> dict:
    score_dict = {}
    for key in module_dict.keys():
        if isinstance(module_dict[key], torch.nn.Conv2d):
            in_ch = module_dict[key].in_channels
            out_ch = module_dict[key].out_channels
            h, w = module_dict[key].kernel_size
            h_add_w = h + w
            h_mul_w = h * w
        else:
            in_ch = module_dict[key].in_features
            out_ch = module_dict[key].out_features
            h_add_w = 0
            h_mul_w = 1

        score = (in_ch + h_add_w + out_ch) / (in_ch * out_ch * h_mul_w)

        weight_not_zero = module_dict[key].weight.data.view(-1) != 0
        weight_all = module_dict[key].weight.data.nelement()
        score *= weight_not_zero

        score *= weight_all / torch.sum(weight_not_zero)

        score_dict.update(
            {key: torch.randn_like(module_dict[key].weight.data.view(-1)).abs() * score}
        )
    return score_dict


def featio(module_dict, cfg_score, train_handler=None) -> dict:
    score_dict = {}
    for key in module_dict.keys():
        featio = 1
        if isinstance(module_dict[key], torch.nn.Conv2d):
            in_ch = module_dict[key].in_channels
            out_ch = module_dict[key].out_channels
            k1, k2 = module_dict[key].kernel_size
            strides, padding = module_dict[key].stride, module_dict[key].padding
            # feati = in_ch * 32 * 32
            # feato = (
            #    out_ch
            #    * ((32 + 2 * padding[0] - k1) // strides[0] + 1)
            #    * ((32 + 2 * padding[1] - k2) // strides[1] + 1)
            # )
            # featio = feato / feati
            featio = out_ch / strides[0] / strides[1] / in_ch
            # featio = min(out_ch / in_ch, in_ch/out_ch)
            # featio = 1 / (in_ch*k1*k2)
        elif isinstance(module_dict[key], torch.nn.Linear):
            in_ch = module_dict[key].in_features
            out_ch = module_dict[key].out_features
            # featio = min(out_ch / in_ch, in_ch/out_ch)
            # featio = 1 / in_ch
            featio = out_ch / in_ch
        else:
            continue
        weight_not_zero = module_dict[key].weight.data.view(-1) != 0
        weight_all = module_dict[key].weight.data.nelement()
        score = (
            torch.randn_like(module_dict[key].weight.data.view(-1)).abs()
            / featio
            * weight_not_zero
        )

        ratio_tmp = torch.sum(weight_not_zero)  # / weight_all
        score /= (ratio_tmp+0.001)  # **2

        score_dict.update({key: score})
    return score_dict


def in_out(module_dict, cfg_score, train_handler=None):
    score_dict = {}
    for key in module_dict.keys():
        if isinstance(module_dict[key], torch.nn.Conv2d):
            score_dict.update(
                {
                    key: torch.randn_like(module_dict[key].weight.data.view(-1)).abs()
                    * (
                        module_dict[key].in_channels / module_dict[key].out_channels
                        + module_dict[key].out_channels / module_dict[key].in_channels
                    )
                    * (module_dict[key].weight.data != 0).view(-1)
                }
            )
        elif isinstance(module_dict[key], torch.nn.Linear):
            score_dict.update(
                {
                    key: torch.randn_like(module_dict[key].weight.data.view(-1)).abs()
                    * (
                        module_dict[key].in_features / module_dict[key].out_features
                        + module_dict[key].out_features / module_dict[key].in_features
                    )
                    * (module_dict[key].weight.data != 0).view(-1)
                }
            )
    return score_dict


def in_out_in(module_dict, cfg_score, train_handler=None):
    score_dict = {}
    for key in module_dict.keys():
        if isinstance(module_dict[key], torch.nn.Conv2d):
            score_dict.update(
                {
                    key: torch.randn_like(module_dict[key].weight.data.view(-1)).abs()
                    * (
                        module_dict[key].in_channels / module_dict[key].out_channels
                        + module_dict[key].out_channels / module_dict[key].in_channels
                    )
                    * (module_dict[key].weight.data != 0).view(-1)
                    / (module_dict[key].in_channels)
                }
            )
        elif isinstance(module_dict[key], torch.nn.Linear):
            score_dict.update(
                {
                    key: torch.randn_like(module_dict[key].weight.data.view(-1)).abs()
                    * (
                        module_dict[key].in_features / module_dict[key].out_features
                        + module_dict[key].out_features / module_dict[key].in_features
                    )
                    * (module_dict[key].weight.data != 0).view(-1)
                    / (module_dict[key].in_features)
                }
            )
    return score_dict


def in_out_ava(module_dict, cfg_score, train_handler=None):
    score_dict = {}
    for key in module_dict.keys():
        weight_not_zero = module_dict[key].weight.data != 0
        if isinstance(module_dict[key], torch.nn.Conv2d):
            score_dict.update(
                {
                    key: torch.randn_like(module_dict[key].weight.data.view(-1)).abs()
                    * (
                        module_dict[key].in_channels / module_dict[key].out_channels
                        + module_dict[key].out_channels / module_dict[key].in_channels
                    )
                    * weight_not_zero.view(-1)
                    / torch.sum(weight_not_zero)
                }
            )
        elif isinstance(module_dict[key], torch.nn.Linear):
            score_dict.update(
                {
                    key: torch.randn_like(module_dict[key].weight.data.view(-1)).abs()
                    * (
                        module_dict[key].in_features / module_dict[key].out_features
                        + module_dict[key].out_features / module_dict[key].in_features
                    )
                    * weight_not_zero.view(-1)
                    / torch.sum(weight_not_zero)
                }
            )
    return score_dict


example_data = None


def snip(module_dict, cfg_score, train_handler=None) -> dict:
    score_dict = {}
    for key in module_dict.keys():
        score_dict.update({key: 0})
    for i, (data, label) in enumerate(train_handler.basic_dict["train_loader"]):
        data = data.to(train_handler.basic_dict["device"])
        label = label.to(train_handler.basic_dict["device"])
        output = train_handler.basic_dict["model"](data)
        torch.nn.functional.cross_entropy(output, label).backward()

        for key in module_dict.keys():
            score_dict[key] += (
                (
                    module_dict[key].weight.data.clone().detach()
                    * module_dict[key].weight.grad.data.clone().detach()
                )
                .view(-1)
                .abs()
            )

            module_dict[key].weight.grad.data.zero_()
        break

    return score_dict


def synflow(module_dict, cfg_score, train_handler=None) -> dict:
    global example_data
    if example_data is None:
        example_data = train_handler.get_example_data()
    score_dict = {}
    train_handler.basic_dict["model"].eval()
    signs = linearize(module_dict)
    input_dim = list(example_data[0, :].shape)
    inputs = torch.ones([1] + input_dim).to(train_handler.basic_dict["device"])
    output = train_handler.basic_dict["model"](inputs)
    torch.sum(output).backward()

    for key in module_dict.keys():
        score_dict.update(
            {
                key: (
                    module_dict[key].weight.data.clone().detach()
                    * module_dict[key].weight.grad.data.clone().detach()
                )
                .view(-1)
                .abs()
            }
        )
        module_dict[key].weight.grad.data.zero_()

    nonlinearize(module_dict, signs)
    train_handler.basic_dict["model"].train()
    return score_dict


def sepeva(module_dict, cfg_score, train_handler=None) -> dict:
    score_dict = {}
    for key in module_dict.keys():
        score_dict.update({key: 0})
    train_handler.basic_dict["model"].eval()

    for key in module_dict.keys():
        input_dim = get_example(module_dict[key])
        inputs = torch.randn_like([1] + input_dim).to(
            train_handler.basic_dict["device"]
        )
        output = module_dict[key](inputs)
        label = torch.zeros_like(output)
        torch.nn.functional.mse_loss(output, label).backward()

        score_dict[key] += (
            (
                module_dict[key].weight.data.clone().detach()
                * module_dict[key].weight.grad.data.clone().detach()
            )
            .view(-1)
            .abs()
        )
        module_dict[key].weight.grad.data.zero_()

    train_handler.basic_dict["model"].train()
    return score_dict


# utils function for synflow
@torch.no_grad()
def linearize(module_dict) -> dict:
    signs = {}
    for key in module_dict.keys():
        signs[key] = module_dict[key].weight.data.view(-1).sign()
        module_dict[key].weight.view(-1).abs_()
    return signs


@torch.no_grad()
def nonlinearize(module_dict, signs):
    for key in module_dict.keys():
        module_dict[key].weight.data.view(-1).mul_(signs[key])


# utils function for synflow_sep
def get_example(module):
    if isinstance(module, torch.nn.Conv2d):
        # a = 1024*4
        # size = int(a/module.in_channels)
        # return [module.in_channels, size, size]
        return [module.in_channels, 32, 32]
    if isinstance(module, torch.nn.Linear):
        return [100, module.in_features]
