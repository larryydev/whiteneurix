import torch
import torch.nn as nn


import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_x):
        x = self.up_conv(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class DeepWhiteBalanceEditing(nn.Module):
    def __init__(self):
        super(DeepWhiteBalanceEditing, self).__init__()
        self.encoder = nn.ModuleList([
            EncoderBlock(3, 24),
            EncoderBlock(24, 48),
            EncoderBlock(48, 96),
            EncoderBlock(96, 192)
        ])

        self.decoder_awb = nn.ModuleList([
            DecoderBlock(192, 192),
            DecoderBlock(192, 96),
            DecoderBlock(96, 48),
            DecoderBlock(48, 24)
        ])

        self.decoder_shade = nn.ModuleList([
            DecoderBlock(192, 192),
            DecoderBlock(192, 96),
            DecoderBlock(96, 48),
            DecoderBlock(48, 24)
        ])

        self.decoder_incandescent = nn.ModuleList([
            DecoderBlock(192, 192),
            DecoderBlock(192, 96),
            DecoderBlock(96, 48),
            DecoderBlock(48, 24)
        ])

        self.final_awb = nn.Conv2d(24, 3, kernel_size=1)
        self.final_shade = nn.Conv2d(24, 3, kernel_size=1)
        self.final_incandescent = nn.Conv2d(24, 3, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)

        awb_output = x
        shade_output = x
        incandescent_output = x

        for i in range(len(self.decoder_awb)):
            awb_output = self.decoder_awb[i](awb_output, skip_connections[-i-1])

        for i in range(len(self.decoder_shade)):
            shade_output = self.decoder_shade[i](shade_output, skip_connections[-i-1])

        for i in range(len(self.decoder_incandescent)):
            incandescent_output = self.decoder_incandescent[i](incandescent_output, skip_connections[-i-1])

        awb_output = self.final_awb(awb_output)
        shade_output = self.final_shade(shade_output)
        incandescent_output = self.final_incandescent(incandescent_output)

        return awb_output, shade_output, incandescent_output