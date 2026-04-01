import torch
import torch.nn as nn

__all__ = ['C3k2_LA']


# -------------------------------
# 基础Conv（兼容YOLO风格）
# -------------------------------
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------------
# DWConv（轻量核心）
# -------------------------------
class DWConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# -------------------------------
# ECA 注意力（轻量）
# -------------------------------
class ECALayer(nn.Module):
    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                          # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


# -------------------------------
# 轻量 Bottleneck
# -------------------------------
class Bottleneck_Lite(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = Conv(c, c, 1, 1)
        self.dw = DWConv(c)

    def forward(self, x):
        return x + self.dw(self.cv1(x))


# -------------------------------
# 增强 Bottleneck（带注意力）
# -------------------------------
class Bottleneck_Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = Conv(c, c, 1, 1)
        self.cv2 = Conv(c, c, 3, 1)
        self.eca = ECALayer(c)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        y = self.eca(y)
        return x + y


# -------------------------------
# ⭐ 最终模块：C3k2_LA
# -------------------------------
class C3k2_LA(nn.Module):
    """
    Lightweight + Attention enhanced C3k2
    Compatible with YOLO26 YAML
    """

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()

        self.c = int(c2 * e)

        # CSP split
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        # Bottleneck stack（非对称设计）
        self.m = nn.ModuleList()
        for i in range(n):
            if i == 0:
                self.m.append(Bottleneck_Lite(self.c))   # 轻量
            else:
                self.m.append(Bottleneck_Attn(self.c))   # 增强

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))