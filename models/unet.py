from einops import rearrange
import torch
from torch import nn
from models.utils import DownSample, Attention, UpSample, Norm, TimeEmbedding, extract
from torch.nn import init
import torch.nn.functional as F


class DDPMTrainObjective(nn.Module):
    def __init__(self, model, beta_1, beta_t, total_diffusion_steps):
        super().__init__()
        self.model = model
        self.total_diffusion_steps = total_diffusion_steps
        self.register_buffer('betas', torch.linspace(beta_1, beta_t, total_diffusion_steps).double())

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """Algorithm 1. from the paper"""
        t = torch.randint(self.total_diffusion_steps, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')

        return loss


class UNet(nn.Module):
    def __init__(self,
                 T: int = 100,  # number of time steps
                 input_channels=3,  # input image channels
                 hid_channels_init=128,
                 ch_mults=(1, 2, 2, 2),
                 attn=(1,),  # layers to apply attention within the resnet block
                 num_res_blocks=2,  # number of residual blocks in each layer
                 dropout=0.):
        super().__init__()

        # Dimensionality
        self.input_channels = input_channels
        self.hid_channels_init = hid_channels_init

        # Sinusoidal Time Embedding
        time_dim = hid_channels_init * 4
        self.time_mlp = TimeEmbedding(T, time_dim)

        # Init conv with padding of 3 (pad input image 28x28 to 32x32; map channels to `hid_channels_init`)
        self.init_conv = nn.Conv2d(self.input_channels, self.hid_channels_init, kernel_size=3, stride=1, padding=1)

        # Define the running channel sizes
        channel_list = [self.hid_channels_init]
        for mult in ch_mults[1:]:
            channel_list.append(channel_list[-1] * mult)
        out_channels = channel_list

        # Down blocks
        self.downs = nn.ModuleList([])
        curr_channel = channel_list[0]
        block_to_ch = [curr_channel]

        for i, ch_out in enumerate(out_channels):
            is_last_layer = i == len(out_channels) - 1

            for j in range(num_res_blocks):
                self.downs.append(ResBlock(
                    in_ch=curr_channel, out_ch=ch_out, time_emb_dim=time_dim,
                    dropout=dropout, attn=(i in attn)))
                curr_channel = ch_out
                block_to_ch.append(curr_channel)

            if not is_last_layer:
                self.downs.append(DownSample(curr_channel))
                block_to_ch.append(curr_channel)

        # Middle Blocks
        self.mid_blocks = nn.ModuleList([  # TODO *2?
            ResBlock(in_ch=curr_channel, out_ch=curr_channel, time_emb_dim=time_dim, dropout=dropout, attn=True),
            ResBlock(in_ch=curr_channel, out_ch=curr_channel, time_emb_dim=time_dim, dropout=dropout, attn=False)
        ])

        # Up blocks
        self.ups = nn.ModuleList([])

        for i, ch_out in reversed(list(enumerate(out_channels))):
            is_first_layer = i == 0

            for j in range(num_res_blocks+1):
                self.ups.append(ResBlock(
                    in_ch=block_to_ch.pop() + curr_channel, out_ch=ch_out, time_emb_dim=time_dim,
                    dropout=dropout, attn=(i in attn)))
                curr_channel = ch_out

            if not is_first_layer:
                self.ups.append(UpSample(curr_channel))

        self.final_block = nn.Sequential(
            nn.GroupNorm(32, curr_channel),
            nn.SiLU(),
            nn.Conv2d(curr_channel, input_channels, 3, stride=1, padding=1)
        )

        self.initialize()

        # Print model size
        print(f'Model size: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M')

    def initialize(self):
        init.xavier_uniform_(self.init_conv.weight)
        init.zeros_(self.init_conv.bias)
        init.xavier_uniform_(self.final_block[-1].weight, gain=1e-5)
        init.zeros_(self.final_block[-1].bias)

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
                analogue_h = hs.pop()
                h = torch.cat([h, analogue_h], dim=1)
            h = up(h, t_emb)

        h = self.final_block(h)

        assert h.shape == x.shape, "Should generate image of the same shape as the input noise (and vice versa)."
        return h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, dropout=0., attn=False):
        super().__init__()

        self.t_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )

        self.residual = nn.Identity()
        self.attn = nn.Identity()

        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

        if attn:
            self.attn = Attention(out_ch)

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