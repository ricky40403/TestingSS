import torch

from torch.nn.modules.batchnorm import _BatchNorm
NORMS = (_BatchNorm)

def get_optimizer( cfg, model):

    param_dict = {}
    optimizer_name = cfg["train"]["optimizer"]["name"]
    optimizer_func = getattr(torch.optim, optimizer_name)

    # # for model in models:
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue
    #     assert param not in param_dict
    #     param_dict[param] = {"name": name}

    # # weight decay of bn is always 0.
    # for name, m in model.named_modules():
    #     if isinstance(m, NORMS):
    #         if hasattr(m, "bias") and m.bias is not None:
    #             param_dict[m.bias].update({"weight_decay": 0})
    #         param_dict[m.weight].update({"weight_decay": 0})

    # # weight decay of bias is always 0.
    # for name, m in model.named_modules():
    #     if hasattr(m, "bias") and m.bias is not None:
    #         param_dict[m.bias].update({"weight_decay": 0})
    # param_groups = []
    # for p, pconfig in param_dict.items():
    #     name = pconfig.pop("name", None)
    #     param_groups += [{"params": p, **pconfig}]


    # optimizer = optimizer(param_groups, **cfg)
    optimizer = optimizer_func(model.parameters(), lr = 0.1)

    return optimizer