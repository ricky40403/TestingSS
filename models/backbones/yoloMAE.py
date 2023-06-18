import torch
import torch.nn as nn

from utils.model_utils import ConvBnAct, CSPBlock

class YoloMAE(nn.Module):
    def __init__(self, cfg, int_c = 3, w_mult = 1.0, h_mult = 1.0):
        super().__init__()

        self.trans_c = 32
        self.trans_h = 3
        self.stage = 5

        self.cn = int(self.trans_c * w_mult)
        self.hn = int(self.trans_h * h_mult)

        self.stem = nn.Sequential(
            ConvBnAct(int_c, self.cn, 3, 1),
            ConvBnAct(self.cn, self.cn * 2, 3, 2),
            ConvBnAct(self.cn * 2, self.cn * 2, 3, 1),
        )

        self.stage2 = nn.Sequential(
            ConvBnAct(self.cn * 2, self.cn * 4, 3, 2),
            CSPBlock(self.cn * 4, self.cn * 4, self.hn, reparam = True)
        )

        self.stage3 = nn.Sequential(
            ConvBnAct(self.cn * 4, self.cn * 8, 3, 2),
            CSPBlock(self.cn * 8, self.cn * 8, self.hn, reparam = True)
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(self.cn * 8, self.cn * 16, 3, 2),
            CSPBlock(self.cn * 16, self.cn * 16, self.hn, reparam = True)
        )

        self.stage5 = nn.Sequential(
            ConvBnAct(self.cn * 16, self.cn * 32, 3, 2),
            CSPBlock(self.cn * 32, self.cn * 32, self.hn, reparam = True)
        )

        self.out_c = [self.cn * 8, self.cn * 16, self.cn * 32]

    def init_weights(self):
        pass

    def get_outC(self):
        return self.out_c
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat1 = x
        x = self.stage4(x)
        feat2 = x        
        x = self.stage5(x)
        feat3 = x
        return feat1, feat2, feat3