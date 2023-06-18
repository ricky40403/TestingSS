import copy

from models.necks.fpn import FPN


def build_neck(cfg):
    model_cfg = copy.deepcopy(cfg)
    name = model_cfg["model"]["neck"]
    if name == 'FPN':
        return FPN(model_cfg)
    else:
        print(f'{name} is not supported yet!')