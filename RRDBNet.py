import functools

import torch
from torch import nn
import torch.nn.functional as F

from RRDB import RRDB
from utils import make_layer


class RRDBNet(nn.Module):
    def __init__(self, in_ch, out_ch, nf, nb, gc=32, conv=nn.Conv2d, padding_mode='zeros', bias=True):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1, bias=bias)

        block = functools.partial(RRDB, nf=nf, gc=gc, conv=conv, padding_mode=padding_mode, bias=bias)

        self.RRDB_trunk = make_layer(block, nb)

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.RRDB_trunk(fea)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        out = self.sig(out)

        return out
