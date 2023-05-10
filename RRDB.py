import torch
from torch import nn

from RD5c import RD5c


class RRDB(nn.Module):
    def __init__(self, nf, gc=32, conv=nn.Conv2d, padding_mode='zeros', bias=True):
        super(RRDB, self).__init__()
        self.RRDB1 = RD5c(nf, gc, bias, conv, padding_mode=padding_mode)
        self.RRDB2 = RD5c(nf, gc, bias, conv, padding_mode=padding_mode)
        self.RRDB3 = RD5c(nf, gc, bias, conv, padding_mode=padding_mode)

    def forward(self, x):
        out = self.RRDB1(x)
        out = self.RRDB2(out)
        out = self.RRDB3(out)
        return out * 0.2 + x
