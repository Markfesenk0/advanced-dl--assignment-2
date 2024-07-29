from einops import rearrange
import torch
from torch import nn
from models.Utils import SinusoidalPosEmb, Downsample, Attention, Upsample, Norm, TimeEmbedding, extract
from torch.nn import init
import torch.nn.functional as F


class UNetTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_t, total_diffusion_steps):
        super().__init__()
        self.model = model
        self.diffusion_steps = total_diffusion_steps
        self.register_buffer('betas', torch.linspace(beta_1, beta_t, total_diffusion_steps).double())

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        t = torch.randint(self.total_diffusion_steps, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')

        return loss


class AttentionUNet(nn.Module):
    def __init__(self, dim=128, dim_mults=(1, 2, 2, 2), channels=3, attn=[1], num_res_blocks=2, dropout=0.):
        super().__init__()

        # Dimensionality
        self.channels = channels
        self.init_dim = dim

        # Sinusoidal Time Embedding
        time_dim = dim * 4
        self.time_mlp = TimeEmbedding(dim, time_dim)

        # Down blocks
        self.init_conv = nn.Conv2d(self.channels, self.init_dim, kernel_size=3, stride=1, padding=1)
        self.downs = nn.ModuleList([])
        dims = [dim]
        cur_dim = dim

        for i, mult in enumerate(dim_mults):
            out_dim = dim * mult

            for j in range(num_res_blocks):
                self.downs.append(ResBlock(
                    dim_in=cur_dim, dim_out=out_dim, time_emb_dim=time_dim,
                    dropout=dropout, attn=(i in attn)))
                cur_dim = out_dim
                dims.append(cur_dim)

            if i != len(dim_mults) - 1:
                self.downs.append(Downsample(cur_dim))
                dims.append(cur_dim)

        # Middle Blocks
        self.mid_blocks = nn.ModuleList([
            ResBlock(dim_in=cur_dim, dim_out=cur_dim, time_emb_dim=time_dim, dropout=dropout, attn=True),
            ResBlock(dim_in=cur_dim, dim_out=cur_dim, time_emb_dim=time_dim, dropout=dropout, attn=False)
        ])

        # Up blocks
        self.ups = nn.ModuleList([])

        for i, mult in reversed(list(enumerate(dim_mults))):
            out_dim = dim * mult

            for j in range(num_res_blocks+1):
                self.ups.append(ResBlock(
                    dim_in=dims.pop() + cur_dim, dim_out=out_dim, time_emb_dim=time_dim,
                    dropout=dropout, attn=(i in attn)))
                cur_dim = out_dim
                dims.append(cur_dim)

            if i != 0:
                self.ups.append(Upsample(cur_dim))
                dims.append(cur_dim)

        self.final_block = nn.Sequential(
            nn.GroupNorm(32, cur_dim),
            nn.SiLU(),
            nn.Conv2d(cur_dim, 3, 3, stride=1, padding=1)
        )

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, time):
        # Sinusoidal Time Embedding
        t_emb = self.time_mlp(time)

        # Downsample
        h = self.init_conv(x)
        hs = [h]
        for down in self.downs:
            h = down(h, t_emb)
            hs.append(h)

        # Middle Blocks
        for mid in self.mid_blocks:
            h = mid(h, t_emb)

        # Upsample
        for up in self.ups:
            if isinstance(up, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = up(h, t_emb)

        return self.final_block(h)


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim=None, dropout=0., attn=False):
        super().__init__()

        self.t_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out),
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, dim_in),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, 3, stride=1, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, dim_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim_out, dim_out, 3, stride=1, padding=1),
        )

        self.residual = nn.Identity()
        self.attn = nn.Identity()

        if dim_in != dim_out:
            self.residual = nn.Conv2d(dim_in, dim_out, 1, stride=1, padding=0)

        if attn:
            self.attn = Attention(dim_out)

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, t):
        h = self.block1(x)
        h += self.t_emb_proj(t)[:, :, None, None]
        h = self.block2(h)

        h = h + self.residual(x)
        h = self.attn(h)

        return h


# class Block(nn.Module):
#     def __init__(self, dim, dim_out, dropout = 0.):
#         super().__init__()
#         self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
#         self.norm = Norm(dim_out)
#         self.act = nn.SiLU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, scale_shift = None):
#         x = self.proj(x)
#         x = self.norm(x)
#
#         if scale_shift:
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift
#
#         x = self.act(x)
#         return self.dropout(x)
#
#
# class ResnetBlock(nn.Module):
#     def __init__(self, dim_in, dim_out, time_emb_dim = None, dropout = 0.):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(time_emb_dim, dim_out * 2)
#         ) if time_emb_dim else None
#
#         self.block1 = Block(dim_in, dim_out, dropout = dropout)
#         self.block2 = Block(dim_out, dim_out)
#         self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
#
#     def forward(self, x, time_emb = None):
#
#         scale_shift = None
#         if self.mlp and time_emb:
#             time_emb = self.mlp(time_emb)
#             time_emb = rearrange(time_emb, 'b c -> b c 1 1')
#             scale_shift = time_emb.chunk(2, dim = 1)
#
#         h = self.block1(x, scale_shift = scale_shift)
#
#         h = self.block2(h)
#
#         return h + self.res_conv(x)