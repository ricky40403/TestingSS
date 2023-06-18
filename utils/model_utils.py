import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import make_divisible



class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)



def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == 'silu':
            # module = SiLU()
            module = nn.SiLU()
        elif name == "swish":
            module = Swish(inplace = inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace = inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace = inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module

    elif isinstance(name, nn.Module):
        return name

    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


def get_norm(name, out_channels):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    elif name == 'gn':
        module = nn.GroupNorm(out_channels)
    else:
        raise NotImplementedError
    return module

class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride=1,
        groups=1,
        bias=False,
        act='silu',
        norm='bn',
        reparam=False,
    ):

        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:            
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x
    

class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1


        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = None
            self.rbr_dense = ConvBnAct(in_channels=in_channels,
                                     out_channels=out_channels,
                                     ksize=kernel_size,
                                     stride=stride,
                                     groups=groups,
                                     act="identity")
            self.rbr_1x1 = ConvBnAct(in_channels=in_channels,
                                   out_channels=out_channels,
                                   ksize=1,
                                   stride=stride,
                                   groups=groups,
                                     act="identity")

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class ResConvBlock(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(ResConvBlock, self).__init__()
        self.stride = stride
        if block_type == 'k1kx':
            self.conv1 = ConvBnAct(in_c, btn_c, 1, stride=1)
        else:
            self.conv1 = ConvBnAct(in_c,
                                  btn_c,
                                  kernel_size,
                                  stride=1)
        if not reparam:
            self.conv2 = ConvBnAct(btn_c, out_c, kernel_size, stride)
        else:
            self.conv2 = RepConv(btn_c,
                                 out_c,
                                 kernel_size,
                                 stride,
                                 act='identity')

        self.activation_function = get_activation(act)

        if in_c != out_c and stride != 2:
            self.residual_proj = ConvBnAct(in_c, out_c, 1, stride=1)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            reslink = self.residual_proj(x)
        else:
            reslink = x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        x = self.activation_function(x)
        return x

class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act='swish',
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBnAct(ch_in, ch_out, k, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y

class CSPBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 n,
                 act='silu',
                 reparam = False,
                 spp=False):
        super(CSPBlock, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = ConvBnAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBnAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_module(
                str(i),
                ResConvBlock(next_ch_in,
                            ch_mid,
                            ch_mid // 2,
                            3,
                            1,
                            reparam=reparam,
                            act=act)
            )
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBnAct(ch_mid * n + ch_first, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y