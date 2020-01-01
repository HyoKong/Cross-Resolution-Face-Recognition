from torch import nn
import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # self.submodule = submodule
        # self.extracted_layers = extracted_layers
    
    def forward(self, x, extracted_layers, submodule):
        outputs = {}
        times = {}
        start = time.time()
        for name, module in submodule._modules.items():
            x = module(x)
            temp = time.time() - start
            if name in extracted_layers:
                outputs[name] = x
                times[name] = temp
        return outputs, times, x, temp


class GroupDepthConv(nn.Module):
    def __init__(self, layer_dict, embedding_dim=512):
        super(GroupDepthConv, self).__init__()
        self.layer_dict = layer_dict
        self.embedding_dim = embedding_dim
        
        for layer, size in layer_dict.items():
            plane = size[1]
            kernel_size = (size[2], size[3])
            setattr(self, layer, ConvBlock(plane, plane, kernel_size, 1, 0, dw=True, linear=False))
            setattr(self, layer + '_linear', ConvBlock(plane, self.embedding_dim, 1, 1, 0, linear=True))
    
    def forward(self, x, layer):
        out = getattr(self, layer)(x)
        out = getattr(self, layer + '_linear')(out)
        out = torch.squeeze(out)
        return out


if __name__ == '__main__':
    c = GroupDepthConv()
