import torch
import torch.nn as nn

from compressai.layers import conv1x1
from einops import rearrange

from model.layers import *
from model.channel import SimulationChannel

__all__ = [
    'SwinJSCC_split',
    'SwinJSCC_slideEnc_ftDec',
]


# Encoder = 13 blocks (B1..B13)
#   B1  : PatchEmbedding + W
#   B2  : SW
#   B3  : PatchMerging  + W
#   B4  : SW
#   B5  : PatchMerging  + W
#   B6  : SW
#   B7  : W
#   B8  : SW
#   B9  : W
#   B10 : SW
#   B11 : PatchMerging  + W
#   B12 : SW
#   B13 : conv1x1 (symbol projection)
#
# Decoder mirrors this layout (PatchReverse / PatchReconstruct fused with the
# preceding SW block), so it also yields 13 blocks (D1..D13).
#
# The channel is selected via `channel_type` ("AWGN" / "Rician"); k_factor is
# ignored for AWGN. This removes the need for parallel `*_Rician` subclasses.
class SwinJSCC_split(nn.Module):
    NUM_BLOCKS = 13

    def __init__(self, SNRdB=10, spp=0.25, img_shape=(256, 256),
                 dims=(128, 192, 256, 320), patch_shape=(2, 2),
                 window_size=8, head_dim=32,
                 channel_type="AWGN", k_factor=1.0, **kwargs):
        super().__init__()

        downscale_factor = 2 ** 4
        total_symbols = img_shape[0] * img_shape[1] * spp
        self.sym_h = img_shape[0] // downscale_factor
        self.sym_w = img_shape[1] // downscale_factor
        self.symbol_dim = int(total_symbols // (self.sym_h * self.sym_w) * 2)

        d0, d1, d2, d3 = dims
        ws, hd = window_size, head_dim

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(                                              # B1
                PatchEmbedding(d0, patch_shape),
                SWAttBlock(d0, d0, hd, ws, 0.0, type='W'),
            ),
            SWAttBlock(d0, d0, hd, ws, 0.0, type='SW'),                 # B2
            nn.Sequential(                                              # B3
                PatchMerging(d0, d1),
                SWAttBlock(d1, d1, hd, ws, 0.0, type='W'),
            ),
            SWAttBlock(d1, d1, hd, ws, 0.0, type='SW'),                 # B4
            nn.Sequential(                                              # B5
                PatchMerging(d1, d2),
                SWAttBlock(d2, d2, hd, ws, 0.0, type='W'),
            ),
            SWAttBlock(d2, d2, hd, ws, 0.0, type='SW'),                 # B6
            SWAttBlock(d2, d2, hd, ws, 0.0, type='W'),                  # B7
            SWAttBlock(d2, d2, hd, ws, 0.0, type='SW'),                 # B8
            SWAttBlock(d2, d2, hd, ws, 0.0, type='W'),                  # B9
            SWAttBlock(d2, d2, hd, ws, 0.0, type='SW'),                 # B10
            nn.Sequential(                                              # B11
                PatchMerging(d2, d3),
                SWAttBlock(d3, d3, hd, ws, 0.0, type='W'),
            ),
            SWAttBlock(d3, d3, hd, ws, 0.0, type='SW'),                 # B12
            conv1x1(d3, self.symbol_dim),                               # B13
        ])

        self.dec_blocks = nn.ModuleList([
            conv1x1(self.symbol_dim, d3),                               # D1
            SWAttBlock(d3, d3, hd, ws, 0.0, type='W'),                  # D2
            nn.Sequential(                                              # D3
                SWAttBlock(d3, d3, hd, ws, 0.0, type='SW'),
                PatchReverse(d3, d2),
            ),
            SWAttBlock(d2, d2, hd, ws, 0.0, type='W'),                  # D4
            SWAttBlock(d2, d2, hd, ws, 0.0, type='SW'),                 # D5
            SWAttBlock(d2, d2, hd, ws, 0.0, type='W'),                  # D6
            SWAttBlock(d2, d2, hd, ws, 0.0, type='SW'),                 # D7
            SWAttBlock(d2, d2, hd, ws, 0.0, type='W'),                  # D8
            nn.Sequential(                                              # D9
                SWAttBlock(d2, d2, hd, ws, 0.0, type='SW'),
                PatchReverse(d2, d1),
            ),
            SWAttBlock(d1, d1, hd, ws, 0.0, type='W'),                  # D10
            nn.Sequential(                                              # D11
                SWAttBlock(d1, d1, hd, ws, 0.0, type='SW'),
                PatchReverse(d1, d0),
            ),
            SWAttBlock(d0, d0, hd, ws, 0.0, type='W'),                  # D12
            nn.Sequential(                                              # D13
                SWAttBlock(d0, d0, hd, ws, 0.0, type='SW'),
                PatchReconstruct(d0, patch_shape),
            ),
        ])

        self.channel = SimulationChannel(
            channel_type=channel_type, SNRdB=SNRdB, k_factor=k_factor
        )

    def forward(self, x):
        # Encoder
        for blk in self.enc_blocks:
            x = blk(x)
        y = x

        # Channel
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2)
        y = torch.complex(y[..., 0], y[..., 1])
        y_hat = self.channel(y)
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w",
                          h=self.sym_h, w=self.sym_w)

        # Decoder
        x_hat = y_hat
        for blk in self.dec_blocks:
            x_hat = blk(x_hat)

        return {"x_hat": x_hat, "y": y}

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dims', type=tuple)
        parser.add_argument('--model.img_shape', type=tuple)
        parser.add_argument('--model.patch_shape', type=tuple)
        parser.add_argument('--model.window_size', type=int)
        parser.add_argument('--model.head_dim', type=int)
        parser.add_argument('--model.channel_type', type=str)
        parser.add_argument('--model.k_factor', type=float)
        return parser


# Sliding-window fine-tuning on top of SwinJSCC_split.
# Unfreezes `num_train_blocks` consecutive encoder blocks starting at
# `start_idx`, and the full decoder. The window must fit within the 13
# encoder blocks, i.e. start_idx + num_train_blocks <= NUM_BLOCKS.
#
# Channel selection inherits from SwinJSCC_split via channel_type kwarg.
class SwinJSCC_slideEnc_ftDec(SwinJSCC_split):
    def __init__(self, exp, start_idx=0, num_train_blocks=4, **kwargs):
        SwinJSCC_split.__init__(self, **kwargs)

        assert start_idx >= 0, f"start_idx must be >= 0, got {start_idx}"
        assert num_train_blocks >= 1, \
            f"num_train_blocks must be >= 1, got {num_train_blocks}"
        assert start_idx + num_train_blocks <= SwinJSCC_split.NUM_BLOCKS, \
            f"start_idx + num_train_blocks must be <= {SwinJSCC_split.NUM_BLOCKS}, " \
            f"got {start_idx} + {num_train_blocks} = {start_idx + num_train_blocks}"
        self.start_idx = start_idx
        self.num_train_blocks = num_train_blocks

        from utils import get_ckpt
        exp_ckpt = get_ckpt(exp)
        pretrained_states = torch.load(exp_ckpt, map_location="cpu")["state_dict"]
        self.load_state_dict(pretrained_states, strict=True)

        # Trainable: encoder window + full decoder
        self.trainable_enc_idx = set(range(start_idx, start_idx + num_train_blocks))

        self.trainable_modules = []
        self.frozen_modules = []
        for i, blk in enumerate(self.enc_blocks):
            if i in self.trainable_enc_idx:
                self.trainable_modules.append(blk)
            else:
                self.frozen_modules.append(blk)
        for blk in self.dec_blocks:
            self.trainable_modules.append(blk)

        for m in self.trainable_modules:
            for p in m.parameters():
                p.requires_grad = True
            m.train()
        for m in self.frozen_modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        for m in self.frozen_modules:
            m.eval()

        for m in self.trainable_modules:
            m.train(mode)

        return self

    def state_dict(self, *args, **kwargs):
        full_state = super().state_dict(*args, **kwargs)

        def is_trainable(k):
            if k.startswith("dec_blocks."):
                return True
            if k.startswith("enc_blocks."):
                idx = int(k.split(".")[1])
                return idx in self.trainable_enc_idx
            return False

        return {k: v for k, v in full_state.items() if is_trainable(k)}

    def load_state_dict(self, state_dict, strict=False, assign=False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.exp', type=str)
        parser.add_argument('--model.start_idx', type=int, default=0)
        parser.add_argument('--model.num_train_blocks', type=int, default=4)
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dims', type=tuple)
        parser.add_argument('--model.img_shape', type=tuple)
        parser.add_argument('--model.patch_shape', type=tuple)
        parser.add_argument('--model.window_size', type=int)
        parser.add_argument('--model.head_dim', type=int)
        parser.add_argument('--model.channel_type', type=str)
        parser.add_argument('--model.k_factor', type=float)
        return parser
