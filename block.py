import torch
import torch.nn as nn


"""
Downscale Block:            maxpool -> double conv
Downscale Bottleneck Block: maxpool -> conv

Upscale Bottleneck Block:   conv -> transpose conv
Upscale Block:              ouble conv -> transpose conv

"""

class EncoderBlock(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_, out_, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.block(x)
        


class DecoderBlock(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_, out_, 
                                          kernel_size=2,
                                          stride=2)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_, out_, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )            

    def forward(self, x, skip_x):
        x = self.up_conv(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.block(x)
        return x
        