import torch.nn as nn
import torch.nn.functional as F


class EncDec(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 256 -> 128
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2) # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16) # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample4 = nn.Upsample(256)  # 128 -> 256
        self.dec_conv4 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.leaky_relu(self.enc_conv0(x)))
        e1 = self.pool1(F.leaky_relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.leaky_relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.leaky_relu(self.enc_conv3(e2)))
        e4 = self.pool4(F.leaky_relu(self.enc_conv4(e3)))

        # bottleneck
        b = F.leaky_relu(self.bottleneck_conv(e4))

        # decoder
        d0 = F.leaky_relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.leaky_relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.leaky_relu(self.dec_conv2(self.upsample2(d1)))
        d3 = F.leaky_relu(self.dec_conv3(self.upsample3(d2)))
        d4 = self.dec_conv4(self.upsample4(d3))  # no activation
        return d4
    