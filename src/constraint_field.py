import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class SinusoidalPosEmb(nn.Module):
    """
    Implements sinusoidal time embeddings to provide the modulation signal.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ModulationBlock(nn.Module):
    """
    Projects time embeddings to the appropriate channel dimension for modulation.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResidualUnit(nn.Module):
    """
    Standard convolutional block with skip connections and modulation signal.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modulation_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # AdaIN-style modulation: project to scale and shift (2 * out_channels)
        self.modulation_proj = nn.Linear(modulation_dim, out_channels * 2)

        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.dropout = nn.Dropout(dropout)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, modulation: torch.Tensor):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        mod = self.modulation_proj(modulation)
        scale, shift = mod.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        h = self.dropout(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for deep layers of the energy landscape mapping.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.q = nn.Conv2d(channels, channels // 8, 1)
        self.k = nn.Conv2d(channels, channels // 8, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        k = self.k(h).view(B, -1, H * W)  # (B, C//8, HW)
        v = self.v(h).view(B, -1, H * W)  # (B, C, HW)

        # q: (B, HW, C//8), k: (B, C//8, HW) -> attn: (B, HW, HW)
        attn = torch.matmul(q, k) * (C // 8) ** -0.5
        attn = F.softmax(attn, dim=-1)

        # attn: (B, HW, HW), v: (B, C, HW) -> out: (B, HW, C)
        out = torch.matmul(attn, v.permute(0, 2, 1))
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(out)


class ConstraintField(nn.Module):
    """
    Constraint Field (U-Net) architecture.
    Maps (B, 1, 28, 28) + time (B,) -> (B, 1, 28, 28).
    Predicts the noise epsilon_theta, which is the gradient of the energy landscape.
    """

    def __init__(
        self,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [14],
        dropout: float = 0.1,
        modulation_dim: int = 256,
    ):
        super().__init__()

        self.time_emb = SinusoidalPosEmb(modulation_dim)
        self.time_mlp = ModulationBlock(modulation_dim, modulation_dim)

        # Encoder
        self.down_blocks = nn.ModuleList()
        curr_channels = 1

        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult

            # Residual blocks
            res_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                res_blocks.append(
                    ResidualUnit(
                        curr_channels if j == 0 else out_channels,
                        out_channels,
                        modulation_dim,
                        dropout,
                    )
                )

            # Attention block
            # Resolution at this level: image_size / (2**i)
            # We assume image_size=28 for MNIST
            res = 28 // (2**i)
            attn = (
                AttentionBlock(out_channels)
                if res in attention_resolutions
                else nn.Identity()
            )

            self.down_blocks.append(
                nn.ModuleDict(
                    {
                        "res": res_blocks,
                        "attn": attn,
                        "down": nn.Conv2d(
                            out_channels, out_channels, 3, stride=2, padding=1
                        )
                        if i < len(channel_multipliers) - 1
                        else nn.Identity(),
                    }
                )
            )
            curr_channels = out_channels

        # Bottleneck
        self.mid_res1 = ResidualUnit(
            curr_channels, curr_channels, modulation_dim, dropout
        )
        self.mid_attn = (
            AttentionBlock(curr_channels)
            if (28 // (2 ** (len(channel_multipliers) - 1))) in attention_resolutions
            else nn.Identity()
        )
        self.mid_res2 = ResidualUnit(
            curr_channels, curr_channels, modulation_dim, dropout
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channel_multipliers))):
            mult = channel_multipliers[i]
            out_channels = base_channels * mult
            in_channels = (
                curr_channels
                if i == len(channel_multipliers) - 1
                else base_channels * channel_multipliers[i + 1]
            )

            # Upsampling
            up = (
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
                if i < len(channel_multipliers) - 1
                else nn.Identity()
            )

            # Residual blocks (input is concat of upsampled and skip)
            res_blocks = nn.ModuleList()

            # The first block in the decoder must handle the concatenated features
            # x has shape (B, out_channels, H, W) after upsampling
            # skip has shape (B, out_channels, H, W)
            # concatenated x has shape (B, 2 * out_channels, H, W)

            # For the very first up-block (deepest level), there's no 'up' ConvTranspose2d,
            # it's an Identity. However, the skip connection from the bottleneck is still there.
            # Wait, in my current implementation:
            # for i in reversed(range(len(channel_multipliers))):
            #   up = ConvTranspose2d if i < len(channel_multipliers) - 1 else Identity
            #   skip = skips.pop()
            #   x = torch.cat([x, skip], dim=1)

            # Let's trace the channels for the first block of each up_block:
            # Bottleneck x: (B, base*mult[2], 7, 7)
            # i=2: up=Identity, skip=(B, base*mult[2], 7, 7) -> cat -> (B, 2*base*mult[2], 7, 7)
            # i=1: up=ConvT(base*mult[2], base*mult[1]), skip=(B, base*mult[1], 14, 14) -> cat -> (B, 2*base*mult[1], 14, 14)
            # i=0: up=ConvT(base*mult[1], base*mult[0]), skip=(B, base*mult[0], 28, 28) -> cat -> (B, 2*base*mult[0], 28, 28)

            # So in EVERY up_block, the first ResidualUnit MUST take 2 * out_channels.

            res_blocks.append(
                ResidualUnit(
                    out_channels * 2,
                    out_channels,
                    modulation_dim,
                    dropout,
                )
            )
            for j in range(num_res_blocks - 1):
                res_blocks.append(
                    ResidualUnit(out_channels, out_channels, modulation_dim, dropout)
                )

            # Attention block
            res = 28 // (2**i)
            attn = (
                AttentionBlock(out_channels)
                if res in attention_resolutions
                else nn.Identity()
            )

            self.up_blocks.append(
                nn.ModuleDict({"up": up, "res": res_blocks, "attn": attn})
            )
            curr_channels = out_channels

        self.final_norm = nn.GroupNorm(min(32, curr_channels), curr_channels)
        self.final_conv = nn.Conv2d(curr_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Modulation projections and time MLP: init to zeros for stable start
                if m.out_features % 2 == 0 and m.out_features >= 128:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Modulation projections output 2*channels (scale+shift), init to zeros for identity
                if m.out_features % 2 == 0 and m.out_features >= 128:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Time embedding
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)

        skips = []

        # Encoder
        for block in self.down_blocks:
            # Cast to nn.ModuleDict to satisfy type checker
            b = block  # type: nn.ModuleDict
            res_blocks: nn.ModuleList = b["res"]
            for res in res_blocks:
                x = res(x, t_emb)
            x = b["attn"](x)
            skips.append(x)
            x = b["down"](x)

        # Bottleneck
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        # Decoder
        for block in self.up_blocks:
            # Cast to nn.ModuleDict to satisfy type checker
            b = block  # type: nn.ModuleDict
            x = b["up"](x)

            # Get skip connection
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1) if skip is not None else x

            res_blocks: nn.ModuleList = b["res"]
            for res in res_blocks:
                x = res(x, t_emb)
            x = b["attn"](x)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x
