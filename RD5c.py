import torch
from torch import nn


class RD5c(nn.Module):

    def __init__(self, nf=64, gc=32, bias=True, conv=nn.Conv2d, padding_mode='zeros'):
        super(RD5c, self).__init__()
        self.conv1 = conv(nf, gc, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        self.conv2 = conv(nf + gc, gc, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        self.conv3 = conv(nf + 2 * gc, gc, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        self.conv4 = conv(nf + 3 * gc, gc, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        self.conv5 = conv(nf + 4 * gc, nf, 3, 1, 1, bias=bias, padding_mode=padding_mode)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x
