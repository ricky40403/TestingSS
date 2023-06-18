import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import ConvBnAct

class FPN(nn.Module):
    def __init__(self, cfg, inc_list = [256, 512, 1024], out_c = 64):
        super(FPN, self).__init__()
        self.cfg = cfg       

        self.neck_count = len(inc_list)
        self.lat_layers = nn.ModuleList()
        self.layeres = nn.ModuleList()      
        for i in range(self.neck_count):
            self.lat_layers.append(
                ConvBnAct(inc_list[i], out_c, 1, 1, norm = None, act="silu")
            )
            self.layeres.append(
                ConvBnAct(out_c, out_c, 3, 1, norm = None, act="silu")
            )      


        self.apply(self.init_conv_kaiming)


    def init_conv_kaiming(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def init_weight(self):
        pass

    
    def up_and_add(self, deep, fine):
        _, _, fine_h, fine_w = [int(x) for x in fine.shape]
        if torch.is_tensor(fine_h):
            fine_h = fine_h.item()
            fine_w = fine_w.item()
        return F.interpolate(deep, size=(fine_h, fine_w),
                    mode='nearest') + fine


    def forward(self, features):
        
        # lat
        FPN_features = []        
        for i in range(len(features)):            
            FPN_features.append(self.lat_layers[i](features[i]))

        #up
        for idx, fpn_feature in enumerate(FPN_features):
            # the deepest feature do nothing                        
            if idx == 0:
                deep_feature = fpn_feature
            else:                
                deep_feature = self.up_and_add(deep_feature, fpn_feature)
                FPN_features[idx] = deep_feature

        
        out_features = []
        # final 3x3 conv for fpn
        for idx in range(len(features)):
            feature = self.layeres[idx](FPN_features[idx])
            out_features.append(feature)

        return out_features