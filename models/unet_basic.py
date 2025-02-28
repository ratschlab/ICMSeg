import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_segmentation import BasicSegmentationModel
from models.switchable_norm import SwitchNorm2d
from typing import Optional

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int]=None, norm_layer: str = 'BN'):
        super().__init__()

        if norm_layer == 'IN':
            normlayer = nn.InstanceNorm2d
        elif norm_layer == 'BN':
            normlayer = nn.BatchNorm2d
        elif norm_layer == 'SN':
            normlayer = SwitchNorm2d
        else:
            raise NotImplementedError

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            normlayer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            normlayer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, norm_layer: str):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_layer=norm_layer)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, norm_layer: str, bilinear: bool = True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_layer=norm_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_layer=norm_layer)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(BasicSegmentationModel):
    def __init__(self, **args):
        super().__init__(**args)
        n_channels = self.hparams.rgb_dim
        n_classes = self.hparams.num_classes
        bilinear = self.hparams.bilinear
        # self.n_channels = n_channels
        layers = [int(l) for l in self.hparams.layers.split('-')]
        if len(layers) != 5:
            print('Current implementation of UNet requires to specify 5 latent layers, e.g. [64, 128, 256, 512, 1024]')
            raise NotImplementedError
        norm_layer = self.hparams.norm_layer
        self.inc = DoubleConv(n_channels, layers[0], norm_layer=norm_layer)
        self.down1 = Down(layers[0], layers[1], norm_layer)
        self.down2 = Down(layers[1], layers[2], norm_layer)
        self.down3 = Down(layers[2], layers[3], norm_layer)
        factor = 2 if bilinear else 1
        self.down4 = Down(layers[3], layers[4] // factor, norm_layer)
        self.up1 = Up(layers[4], layers[3] // factor, norm_layer, bilinear)
        self.up2 = Up(layers[3], layers[2] // factor, norm_layer, bilinear)
        self.up3 = Up(layers[2], layers[1] // factor, norm_layer, bilinear)
        self.up4 = Up(layers[1], layers[0], norm_layer, bilinear)
        self.outc = OutConv(layers[0], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        latent = self.up3(x, x2)
        x = self.up4(latent, x1)
        logits = self.outc(x)
       
        return logits