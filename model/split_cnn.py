import torch
import torch.nn as nn

from compressai.layers import conv1x1
from compressai.models.utils import conv, deconv
from einops import rearrange

from model.channel import SimulationChannel
from model.layers import resNetBlock

__all__ = [
    'SwinJSCC_split_CNN',
]


class SwinJSCC_split_CNN(nn.Module):
    """Block-wise version of :class:`SwinJSCC_CNN`.

    The encoder and decoder are each divided into 13 blocks using the same
    boundaries as ``SwinJSCC_split``.  The split makes individual blocks easy
    to freeze or fine-tune without changing the underlying computation.
    """

    NUM_BLOCKS = 13

    def __init__(self, SNRdB=10, spp=0.25, img_shape=(256, 256), dim=384,
                 channel_type="AWGN", k_factor=1.0, **kwargs):
        super().__init__()

        downscale_factor = 2 ** 4
        total_symbols = img_shape[0] * img_shape[1] * spp
        self.sym_h = img_shape[0] // downscale_factor
        self.sym_w = img_shape[1] // downscale_factor
        self.symbol_dim = int(
            total_symbols // (self.sym_h * self.sym_w) * 2
        )

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(                                      # B1
                conv(3, dim),
                resNetBlock(dim),
            ),
            resNetBlock(dim),                                   # B2
            nn.Sequential(                                      # B3
                conv(dim, dim),
                resNetBlock(dim),
            ),
            resNetBlock(dim),                                   # B4
            nn.Sequential(                                      # B5
                conv(dim, 2 * dim),
                resNetBlock(2 * dim),
            ),
            resNetBlock(2 * dim),                               # B6
            resNetBlock(2 * dim),                               # B7
            resNetBlock(2 * dim),                               # B8
            resNetBlock(2 * dim),                               # B9
            resNetBlock(2 * dim),                               # B10
            nn.Sequential(                                      # B11
                conv(2 * dim, 2 * dim),
                resNetBlock(2 * dim),
            ),
            resNetBlock(2 * dim),                               # B12
            conv1x1(2 * dim, self.symbol_dim),                  # B13
        ])

        self.dec_blocks = nn.ModuleList([
            conv1x1(self.symbol_dim, 2 * dim),                  # D1
            resNetBlock(2 * dim),                               # D2
            nn.Sequential(                                      # D3
                resNetBlock(2 * dim),
                deconv(2 * dim, 2 * dim),
            ),
            resNetBlock(2 * dim),                               # D4
            resNetBlock(2 * dim),                               # D5
            resNetBlock(2 * dim),                               # D6
            resNetBlock(2 * dim),                               # D7
            resNetBlock(2 * dim),                               # D8
            nn.Sequential(                                      # D9
                resNetBlock(2 * dim),
                deconv(2 * dim, dim),
            ),
            resNetBlock(dim),                                   # D10
            nn.Sequential(                                      # D11
                resNetBlock(dim),
                deconv(dim, dim),
            ),
            resNetBlock(dim),                                   # D12
            nn.Sequential(                                      # D13
                resNetBlock(dim),
                deconv(dim, 3),
            ),
        ])

        self.channel = SimulationChannel(
            channel_type=channel_type, SNRdB=SNRdB, k_factor=k_factor
        )

    def forward(self, x):
        for blk in self.enc_blocks:
            x = blk(x)
        y = x

        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2)
        y = torch.complex(y[..., 0], y[..., 1])
        y_hat = self.channel(y)
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(
            y_hat,
            "b (n_c h w) iq -> b (n_c iq) h w",
            h=self.sym_h,
            w=self.sym_w,
        )

        x_hat = y_hat
        for blk in self.dec_blocks:
            x_hat = blk(x_hat)

        return {"x_hat": x_hat, "y": y}

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        parser.add_argument('--model.channel_type', type=str)
        parser.add_argument('--model.k_factor', type=float)
        return parser
