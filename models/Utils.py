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
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

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
    def __init__(self, dim, time_dim, sinusoidal_pos_emb_theta=10000):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
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


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


def extract(v, t, x_shape):
    """
    Extract coefficients at specified timesteps and
    reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

# class Attention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
#         super().__init__()
#         self.heads = heads
#         hidden_dim = dim_head * heads
#
#         self.norm = Norm(dim)
#
#         self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         x = self.norm(x)
#
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)
#
#         mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
#         k, v = map(lambda t: torch.cat(t, dim=-2), ((mk, k), (mv, v)))
#
#         # similarity
#
#         scale = q.shape[-1] ** -0.5
#         sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
#
#         # attention
#
#         attn = sim.softmax(dim=-1)
#         attn = self.attn_dropout(attn)
#
#         out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
#
#         out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
#         return self.to_out(out)
