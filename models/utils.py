import math
from einops.layers.torch import Rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, dim)
        self.proj_q = nn.Conv2d(dim, dim, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(dim, dim, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(dim, dim, 1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(dim, dim, 1, stride=1, padding=0)

        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj_out(h)

        return x + h


class TimeEmbedding(nn.Module):
    def __init__(self, T, time_dim):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(T),
            nn.Linear(T, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.time_emb(t)
        return emb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


def default(val, d):
    if val:
        return val
    return d


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb=None):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb=None):
        x = self.main(x)
        return x