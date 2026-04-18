import torch
import torch.nn as nn


class DiffusionUNet(nn.Module):
    """Diffusion UNet with time embedding - matches your trained model"""

    def __init__(self, in_channels=6, time_emb_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder with time embeddings
        self.down1 = self._down_block(in_channels, 64, time_emb_dim)
        self.down2 = self._down_block(64, 128, time_emb_dim)
        self.down3 = self._down_block(128, 256, time_emb_dim)
        self.down4 = self._down_block(256, 512, time_emb_dim)

        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
        )

        # Decoder with skip connections and time embeddings
        self.up4 = self._up_block(512 + 512, 256, time_emb_dim)
        self.up3 = self._up_block(256 + 256, 128, time_emb_dim)
        self.up2 = self._up_block(128 + 128, 64, time_emb_dim)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 3, 4, 2, 1),
            nn.Sigmoid()  # CRITICAL: Constrain output to [0, 1]
        )

    def _down_block(self, in_ch, out_ch, time_emb_dim):
        return nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ),
            'time_emb': nn.Linear(time_emb_dim, out_ch)
        })

    def _up_block(self, in_ch, out_ch, time_emb_dim):
        return nn.ModuleDict({
            'conv': nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ),
            'time_emb': nn.Linear(time_emb_dim, out_ch)
        })

    def forward(self, x, t=None):
        # If no timestep provided, use 0 (for inference)
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)

        # Normalize time to [0, 1]
        t = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_mlp(t)

        # Encoder
        d1 = self.down1['conv'](x)
        d1 = d1 + self.down1['time_emb'](t_emb)[:, :, None, None]

        d2 = self.down2['conv'](d1)
        d2 = d2 + self.down2['time_emb'](t_emb)[:, :, None, None]

        d3 = self.down3['conv'](d2)
        d3 = d3 + self.down3['time_emb'](t_emb)[:, :, None, None]

        d4 = self.down4['conv'](d3)
        d4 = d4 + self.down4['time_emb'](t_emb)[:, :, None, None]

        # Bottleneck
        mid = self.mid(d4)

        # Decoder
        u4 = self.up4['conv'](torch.cat([mid, d4], dim=1))
        u4 = u4 + self.up4['time_emb'](t_emb)[:, :, None, None]

        u3 = self.up3['conv'](torch.cat([u4, d3], dim=1))
        u3 = u3 + self.up3['time_emb'](t_emb)[:, :, None, None]

        u2 = self.up2['conv'](torch.cat([u3, d2], dim=1))
        u2 = u2 + self.up2['time_emb'](t_emb)[:, :, None, None]

        out = self.up1(torch.cat([u2, d1], dim=1))
        return out