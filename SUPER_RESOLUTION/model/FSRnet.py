import torch
import numpy as np

import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

import pdb


class _Residual_Block(nn.Module):
    def __init__(self, out_channels, in_channels=64):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.conv1 = OctConv(ch_in=in_channels,ch_out=out_channels,kernel_size=3,stride=1,alphas=(0,0.5))
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)
        # self.batch_channel = int(160*out_channels/64)
        # self.in1 = nn.InstanceNorm2d(self.batch_channel, affine=True)
        self.relu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.conv2 = OctConv(ch_in=in_channels, ch_out=out_channels, kernel_size=3, stride=1, alphas=(0.5, 0))
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity_data = x
        out = self.conv1(x)
        out = self.relu(self.in1(out))
        out = self.conv2(out)
        out = self.in2(out)
        out = torch.add(out, identity_data)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes * 2)
        self.relu = nn.PReLU(128)
        self.conv2 = conv3x3(planes * 2, planes * 2)
        self.bn2 = nn.InstanceNorm2d(planes * 2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.InstanceNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''

    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 128
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.InstanceNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.InstanceNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
            conv,
            bn,
            self.relu,
        )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out


def hg2(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=2, num_blocks=4,
                         num_classes=kwargs['num_classes'])
    return model


class Coarse_SR_Network(nn.Module):
    def __init__(self, ngf=64, n_blocks=6):
        super(Coarse_SR_Network, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=7, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReLU(True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, padding=0, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 2, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, padding=0, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True)
                  ]

        for i in range(n_blocks):
            model += [_Residual_Block(out_channels=ngf * 4)]

        model += [nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, ngf, 3, 1, 0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReLU(True),
                  ]
        self.model = nn.Sequential(*model)

        out = []
        out += [nn.ReflectionPad2d(1),
                nn.Conv2d(ngf, 3, 3, 1, 0, bias=False),
                nn.Tanh()]
        self.out = nn.Sequential(*out)

    def forward(self, x):
        coarse_feature = self.model(x)
        coarse_img = self.out(coarse_feature)
        return coarse_img, coarse_feature


class Fine_SR_Encoder(nn.Module):
    def __init__(self, ngf=64, n_blocks=6):
        '''
        :param ngf:
        :param n_blocks:
        '''
        super(Fine_SR_Encoder, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, padding=0, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 2, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, padding=0, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True)
                  ]
        for i in range(n_blocks):
            model += [_Residual_Block(out_channels=ngf * 4)]

        model += [nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, ngf, 3, 1, 0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReLU(True),
                  ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Prior_Estimation_Network(nn.Module):
    def __init__(self, n_hourglass=4, n_blocks=2, ngf=64, parsing_classes=13, num_landmark=68):
        super(Prior_Estimation_Network, self).__init__()
        self.fc = nn.Conv2d(in_channels=128, out_channels=parsing_classes, kernel_size=1, bias=True)
        self.fc_landmark = nn.Conv2d(in_channels=128, out_channels=num_landmark, kernel_size=1, bias=False)

        model = []
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=7, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReLU(True)
                  ]

        for i in range(n_blocks):
            model += [_Residual_Block(ngf)]

        for i in range(n_hourglass):
            model += [Hourglass(planes=ngf, depth=4, block=BasicBlock, num_blocks=3)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        parsing_out = self.fc(out)
        landmark_out = self.fc_landmark(out)
        return out, landmark_out, parsing_out


class Fine_SR_Decoder(nn.Module):
    def __init__(self, ngf=128, n_blocks=6):
        super(Fine_SR_Decoder, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=2, padding=0, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 2, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=2, padding=0, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=3, stride=1, padding=0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True)
                  ]

        for i in range(n_blocks):
            model += [_Residual_Block(out_channels=ngf * 4)]

        model += [nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 0, bias=False),
                  nn.InstanceNorm2d(ngf * 2),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1, bias=False),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(ngf, ngf, 3, 1, 0, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReLU(True),
                  ]

        out = []
        out += [nn.ReflectionPad2d(1),
                nn.Conv2d(ngf, 3, 3, 1, 0, bias=False),
                nn.Tanh()]
        self.out = nn.Sequential(*out)

    def forward(self, x):
        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.PReLU(64)
        self.bn_mid = nn.BatchNorm2d(64, affine=True)
        self.residual = self.make_layer(_Residual_Block, 3, out_channel=64, in_channel=64)
        self.fc = nn.Linear(64 * 56 * 56, 512)
        self.bn_end = nn.BatchNorm1d(512)

    def make_layer(self, block, num_of_layer, in_channel, out_channel):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(out_channel, in_channels=in_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn_mid(self.conv_input(x)))
        # out = self.residual(out)
        # out = self.residual(out)
        out = self.bn_mid(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.bn_end(out)
        return out


class OverallNetwork(nn.Module):
    def __init__(self):
        super(OverallNetwork, self).__init__()
        self._coarse_sr_network = Course_SR_Network()
        self._prior_estimation_network = Prior_Estimation_Network()
        self._fine_sr_encoder = Fine_SR_Encoder()
        self._fine_sr_decoder = Fine_SR_Decoder()
        self.softmax = nn.Softmax()
        # self.deconv = nn.ConvTranspose2d(in_channels=16,out_channels=11,kernel_size=3,stride=2,bias=False,padding=1,output_padding=1)

    def forward(self, x):
        out, coarse_out = self._coarse_sr_network(x)
        out_sr = self._fine_sr_encoder(out)
        out_pe, landmark_out, parsing_out = self._prior_estimation_network(out)
        # landmark_out = self.softmax(landmark_out)
        # parsing_out = self.deconv(parsing_out)
        # out = torch.cat((out_sr,landmark_out),1)
        # out = torch.cat((out,parsing_out),1)
        out = torch.cat((out_pe, out_sr), 1)
        out = self._fine_sr_decoder(out)
        # pdb.set_trace()
        return coarse_out, out, landmark_out, parsing_out


class OverallNetwork_GAN(nn.Module):
    def __init__(self):
        super(OverallNetwork_GAN, self).__init__()
        self._coarse_sr_network = Course_SR_Network()
        self._prior_estimation_network = Prior_Estimation_Network()
        self._fine_sr_encoder = Fine_SR_Encoder()
        self._fine_sr_decoder = Fine_SR_Decoder()
        self._discriminator = Discriminator()
        # self._first = nn.Sequential(
        #     self._coarse_sr_network(),
        #     self._fine_sr_encoder(),
        #     self._prior_estimation_network()
        # )
        # self.deconv = nn.ConvTranspose2d(in_channels=16,out_channels=11,kernel_size=3,stride=2,bias=False,padding=1,output_padding=1)

    def forward_once(self, x):
        # out,coarse_out,out_sr,out_pe,landmark_out,parsing_out = self._first(x)
        # out,coarse_out = self._coarse_sr_network(x)
        out_sr = self._fine_sr_encoder(x)
        out_pe, landmark_out, parsing_out = self._prior_estimation_network(x)
        # landmark_out = self.softmax(landmark_out)
        # parsing_out = self.deconv(parsing_out)
        # out = torch.cat((out_sr,landmark_out),1)
        # out = torch.cat((out,parsing_out),1)
        out = torch.cat((out_pe, out_sr), 1)
        criterion_out = self._discriminator(out)
        # pdb.set_trace()
        return out, landmark_out, parsing_out, criterion_out

    def forward(self, lr, hr):
        out, coarse = self._coarse_sr_network(lr)
        out1, landmark_out1, parsing_out1, embedding1 = self.forward_once(coarse)
        out2, landmark_out2, parsing_out2, embedding2 = self.forward_once(hr)

        sr = self._fine_sr_decoder(out1)

        return sr, coarse, landmark_out1, parsing_out1, embedding1, embedding2


if __name__ == '__main__':
    model = OverallNetwork().cuda()
    input_data = Variable(torch.rand(3, 3, 224, 224)).cuda()

    print(model(input_data)[0].size())
