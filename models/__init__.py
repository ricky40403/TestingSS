
from models.SSHand import SSHand






def build_models(cfg):
    model = None
    if cfg["model"]["name"] == "SSHand":
        model = SSHand(cfg)

    return model