import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CSCB']


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ECALayer(nn.Module):
    """Extremely lightweight channel attention"""
    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class CSCB(nn.Module):
    """
    CSCB: Cross-Scale Channel Branch Downsampling Block
    - YOLO26 compatible
    - lightweight
    - small-object friendly
    """

    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()
        assert c2 % 2 == 0

        # -------- Channel split --------
        c_main = int(c2 * 0.75)
        c_aux = c2 - c_main

        # -------- Main branch --------
        self.cv1 = nn.Sequential(
            Conv(c1 * 3 // 4, c1 * 3 // 4, 3, 2, g=c1 * 3 // 4),  # DWConv
            Conv(c1 * 3 // 4, c_main, 1, 1)                      # PWConv
        )

        # -------- Aux branch --------
        self.cv2 = Conv(c1 // 4, c_aux, 1, 1)

        # -------- Channel alignment (关键修复) --------
        self.align_aux2main = nn.Conv2d(c_aux, c_main, 1, bias=False)

        # -------- Cross-branch gating --------
        self.alpha = nn.Parameter(torch.zeros(1))
        # ❗去掉 beta（避免再次不匹配 + 提高稳定性）

        # -------- Attention --------
        self.eca = ECALayer(c2)
        self.spatial = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

    def forward(self, x):

        # ---- Downsample ----
        x = F.avg_pool2d(x, 2, 1)

        # ---- Channel split (3:1) ----
        c = x.shape[1]
        x1 = x[:, :c * 3 // 4]
        x2 = x[:, c * 3 // 4:]

        # ---- Branch forward ----
        y1 = self.cv1(x1)
        y2 = self.cv2(F.max_pool2d(x2, 3, 2, 1))

        # ---- Cross interaction（修复后）----
        y1 = y1 + self.alpha * self.align_aux2main(y2)

        # ---- Concat ----
        out = torch.cat((y1, y2), 1)

        # ---- Channel attention ----
        out = self.eca(out)

        # ---- Spatial attention ----
        s = torch.mean(out, dim=1, keepdim=True)
        s = torch.sigmoid(self.spatial(s))
        out = out * s

        return out