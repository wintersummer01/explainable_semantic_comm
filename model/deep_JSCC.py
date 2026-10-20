import torch
import torch.nn as nn

from einops import rearrange
from compressai.layers import conv1x1
from compressai.models.utils import conv, deconv

from model.layers import *
from model.channel import AWGNChannel, PseudoRicianChannel

__all__ = ['DeepJSCC', 'DeepJSCC_CNN', 'SwinJSCC', 'SwinJSCC_Rician', 'SwinJSCC_CNN']


class DeepJSCC(nn.Module):
    def __init__(self, SNRdB=10, spp=0.25, img_shape=(256, 256), dim=384, patch_shape=(4, 4), window_size=8, base_num_head=8, **kwargs):
        super(DeepJSCC, self).__init__()
        
        downscale_factor = 2 ** 4
        total_symbols = img_shape[0] * img_shape[1] * spp # symbols = pixels * spp
        self.sym_h = img_shape[0] // downscale_factor
        self.sym_w = img_shape[1] // downscale_factor
        symbol_dim = int(total_symbols // (self.sym_h * self.sym_w) * 2) # Real and Imag : *2
        
        head_dim = dim // base_num_head
        
        self.encoder = nn.Sequential(
            PatchEmbedding(dim, patch_shape),
            SWAttBlock(dim, dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dim, dim, head_dim, window_size, 0.0, type='SW'),
            PatchMerging(dim, 2*dim),
            SWAttBlock(2*dim, 2*dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(2*dim, 2*dim, head_dim, window_size, 0.0, type='SW'),
            PatchMerging(2*dim, 4*dim),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='SW'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='SW'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='SW'),
            conv1x1(4*dim, symbol_dim),
        )
        self.decoder = nn.Sequential(
            conv1x1(symbol_dim, 4*dim),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='SW'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='SW'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(4*dim, 4*dim, head_dim, window_size, 0.0, type='SW'),
            PatchReverse(4*dim, 2*dim),
            SWAttBlock(2*dim, 2*dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(2*dim, 2*dim, head_dim, window_size, 0.0, type='SW'),
            PatchReverse(2*dim, dim),
            SWAttBlock(dim, dim, head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dim, dim, head_dim, window_size, 0.0, type='SW'),
            PatchReconstruct(dim, patch_shape)
        )
        self.channel = AWGNChannel(SNRdB=SNRdB)
        
    def forward(self, x):
        y = self.encoder(x)
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2)
        y = torch.complex(y[..., 0], y[..., 1])
        
        y_hat = self.channel(y)
        
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=self.sym_h, w=self.sym_w)
        x_hat = self.decoder(y_hat)

        # return x_hat
        return {
            "x_hat": x_hat, "y":y
        }
    
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        parser.add_argument('--model.patch_shape', type=tuple)
        parser.add_argument('--model.window_size', type=int) 
        parser.add_argument('--model.base_num_head', type=int) 
        
        return parser
   
class DeepJSCC_CNN(DeepJSCC):
    def __init__(self, SNRdB=10, spp=0.25, img_shape=(256, 256), dim=384, **kwargs):
        super().__init__()
        
        downscale_factor = 2 ** 4
        total_symbols = img_shape[0] * img_shape[1] * spp # symbols = pixels * spp
        self.sym_h = img_shape[0] // downscale_factor
        self.sym_w = img_shape[1] // downscale_factor
        symbol_dim = int(total_symbols // (self.sym_h * self.sym_w) * 2) # Real and Imag : *2
        
        self.encoder = nn.Sequential(
            conv(3, dim),
            resNetBlock(dim),
            resNetBlock(dim),
            conv(dim, dim),
            resNetBlock(dim),
            resNetBlock(dim),
            conv(dim, 2*dim),
            resNetBlock(2*dim),
            resNetBlock(2*dim),
            conv(2*dim, symbol_dim),
        )
        self.decoder = nn.Sequential(
            deconv(symbol_dim, 2*dim),
            resNetBlock(2*dim),
            resNetBlock(2*dim),
            deconv(2*dim, dim),
            resNetBlock(dim),
            resNetBlock(dim),
            deconv(dim, dim),
            resNetBlock(dim),
            resNetBlock(dim),
            deconv(dim, 3),
        )
        self.channel = AWGNChannel(SNRdB=SNRdB)
        
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        
        return parser



class SwinJSCC(nn.Module):
    def __init__(self, SNRdB=10, spp=0.25, img_shape=(256, 256), dims=(128, 192, 256, 320), patch_shape=(2, 2), window_size=8, head_dim=32, **kwargs):
        super(SwinJSCC, self).__init__()
        
        downscale_factor = 2 ** 4
        total_symbols = img_shape[0] * img_shape[1] * spp # symbols = pixels * spp
        self.sym_h = img_shape[0] // downscale_factor
        self.sym_w = img_shape[1] // downscale_factor
        self.symbol_dim = int(total_symbols // (self.sym_h * self.sym_w) * 2) # Real and Imag : *2
        
        self.enc_1 = nn.Sequential(
            PatchEmbedding(dims[0], patch_shape), # 2 x 2 for this model
            SWAttBlock(dims[0], dims[0], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[0], dims[0], head_dim, window_size, 0.0, type='SW'),
        )
        self.enc_2 = nn.Sequential(
            PatchMerging(dims[0], dims[1]),
            SWAttBlock(dims[1], dims[1], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[1], dims[1], head_dim, window_size, 0.0, type='SW'),
        )
        self.enc_3_0 =  nn.Sequential(
            PatchMerging(dims[1], dims[2]),
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='SW'),
        )
        self.enc_3_1 =  nn.Sequential(
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='SW'),
        )
        self.enc_3_2 =  nn.Sequential(
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='SW'),
        )
        self.enc_4 = nn.Sequential(
            PatchMerging(dims[2], dims[3]),
            SWAttBlock(dims[3], dims[3], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[3], dims[3], head_dim, window_size, 0.0, type='SW'),
        )
        self.enc_proj = conv1x1(dims[3], self.symbol_dim)
        
        
        self.dec_proj = conv1x1(self.symbol_dim, dims[3])
        self.dec_1 = nn.Sequential(
            SWAttBlock(dims[3], dims[3], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[3], dims[3], head_dim, window_size, 0.0, type='SW'),
            PatchReverse(dims[3], dims[2]),
        )
        self.dec_2_0 =  nn.Sequential(
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='SW'),
        )
        self.dec_2_1 =  nn.Sequential(
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='SW'),
        )
        self.dec_2_2 =  nn.Sequential(
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[2], dims[2], head_dim, window_size, 0.0, type='SW'),
            PatchReverse(dims[2], dims[1]),
        )
        self.dec_3 = nn.Sequential(
            SWAttBlock(dims[1], dims[1], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[1], dims[1], head_dim, window_size, 0.0, type='SW'),
            PatchReverse(dims[1], dims[0]),
        )
        self.dec_4 = nn.Sequential(
            SWAttBlock(dims[0], dims[0], head_dim, window_size, 0.0, type='W'),
            SWAttBlock(dims[0], dims[0], head_dim, window_size, 0.0, type='SW'),
            PatchReconstruct(dims[0], patch_shape)
        )
        
        self.channel = AWGNChannel(SNRdB=SNRdB)
        
    def forward(self, x):
        # Encoder
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3_0(x)
        x = self.enc_3_1(x)
        x = self.enc_3_2(x)
        x = self.enc_4(x)
        y = self.enc_proj(x)
        
        # Channel
        y = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2)
        y = torch.complex(y[..., 0], y[..., 1])
        y_hat = self.channel(y)
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=self.sym_h, w=self.sym_w)
        
        # Decoder
        x_hat = self.dec_proj(y_hat)
        x_hat = self.dec_1(x_hat)
        x_hat = self.dec_2_0(x_hat)
        x_hat = self.dec_2_1(x_hat)
        x_hat = self.dec_2_2(x_hat)
        x_hat = self.dec_3(x_hat)
        x_hat = self.dec_4(x_hat)

        # return x_hat
        return {
            "x_hat": x_hat, "y":y
        }
    
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dims', type=tuple)
        parser.add_argument('--model.img_shape', type=tuple)
        parser.add_argument('--model.patch_shape', type=tuple)
        parser.add_argument('--model.window_size', type=int) 
        parser.add_argument('--model.head_dim', type=int) 
        
        return parser


class SwinJSCC_Rician(SwinJSCC):
    def __init__(self, k_factor, SNRdB=10, **kwargs):
        super().__init__(SNRdB=SNRdB, **kwargs)
        self.channel = PseudoRicianChannel(K=k_factor, SNRdB=SNRdB)
        
    @staticmethod
    def get_parser(parser):
        parser = SwinJSCC.get_parser(parser)
        parser.add_argument('--model.k_factor', type=float) 
        return parser

   
class SwinJSCC_CNN(SwinJSCC):
    def __init__(self, SNRdB=10, spp=0.25, img_shape=(256, 256), dim=384, **kwargs):
        # Initialize nn.Module directly: all Swin-specific modules are replaced
        # below, while SwinJSCC.forward is intentionally reused.
        nn.Module.__init__(self)

        downscale_factor = 2 ** 4
        total_symbols = img_shape[0] * img_shape[1] * spp
        self.sym_h = img_shape[0] // downscale_factor
        self.sym_w = img_shape[1] // downscale_factor
        self.symbol_dim = int(
            total_symbols // (self.sym_h * self.sym_w) * 2
        )  # real and imaginary parts

        # Match SwinJSCC's 2/2/6/2 encoder block layout.  Keeping the same
        # stage names also makes this model compatible with code that inspects
        # intermediate SwinJSCC activations.
        self.enc_1 = nn.Sequential(
            conv(3, dim),
            resNetBlock(dim),
            resNetBlock(dim),
        )
        self.enc_2 = nn.Sequential(
            conv(dim, dim),
            resNetBlock(dim),
            resNetBlock(dim),
        )
        self.enc_3_0 = nn.Sequential(
            conv(dim, 2 * dim),
            resNetBlock(2 * dim),
            resNetBlock(2 * dim),
        )
        self.enc_3_1 = nn.Sequential(
            resNetBlock(2 * dim),
            resNetBlock(2 * dim),
        )
        self.enc_3_2 = nn.Sequential(
            resNetBlock(2 * dim),
            resNetBlock(2 * dim),
        )
        self.enc_4 = nn.Sequential(
            conv(2 * dim, 2 * dim),
            resNetBlock(2 * dim),
            resNetBlock(2 * dim),
        )
        self.enc_proj = conv1x1(2 * dim, self.symbol_dim)

        self.dec_proj = conv1x1(self.symbol_dim, 2 * dim)
        self.dec_1 = nn.Sequential(
            resNetBlock(2 * dim),
            resNetBlock(2 * dim),
            deconv(2 * dim, 2 * dim),
        )
        self.dec_2_0 = nn.Sequential(
            resNetBlock(2 * dim),
            resNetBlock(2 * dim),
        )
        self.dec_2_1 = nn.Sequential(
            resNetBlock(2 * dim),
            resNetBlock(2 * dim),
        )
        self.dec_2_2 = nn.Sequential(
            resNetBlock(2 * dim),
            resNetBlock(2 * dim),
            deconv(2 * dim, dim),
        )
        self.dec_3 = nn.Sequential(
            resNetBlock(dim),
            resNetBlock(dim),
            deconv(dim, dim),
        )
        self.dec_4 = nn.Sequential(
            resNetBlock(dim),
            resNetBlock(dim),
            deconv(dim, 3),
        )

        self.channel = AWGNChannel(SNRdB=SNRdB)

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)

        return parser
