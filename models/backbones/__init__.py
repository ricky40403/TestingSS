import copy
from models.backbones.yoloMAE import YoloMAE



def build_backbone(cfg):
    model_cfg = copy.deepcopy(cfg)
    name = model_cfg["model"]["backbone"]
    if name == 'YoloMAE':
        return YoloMAE(model_cfg)
    else:
        print(f'{name} is not supported yet!')