import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['AtD','LCM']
class BlurPool(nn.Module):

    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride = stride

        # 3x3 Gaussian-like kernel
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]])
        kernel = kernel / kernel.sum()

        self.register_buffer('kernel', kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.groups = channels

    def forward(self, x):
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=self.groups)
        return x


class AtD(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blur = BlurPool(in_channels, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.blur(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class LCM(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.conv1 = nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        avg = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        maxv = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        contrast = maxv - avg

        out = torch.cat([x, contrast], dim=1)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return x + out   # residual