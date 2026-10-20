import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from compressai.layers import conv3x3, conv1x1
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange
from einops.layers.torch import Rearrange

__all__ = [
    "resNetBlock",
    "SWAttBlock",
    "PatchEmbedding",
    "PatchReconstruct",
    "PatchMerging",
    "PatchReverse"
]

class resNetBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out
    
class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class SWAttBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(SWAttBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = Rearrange('b c h w -> b h w c')(x)
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        x = Rearrange('b h w c -> b c h w')(x)
        return x
    

    
class PatchEmbedding(nn.Module):
    def __init__(self, out_dim, patch_size=[4,4], in_dim=3):
        """ Patch embedding block
        """
        super(PatchEmbedding, self).__init__()
        self.patch_div = nn.Conv2d(
            in_dim, out_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
        )
        self.ln = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        x = self.patch_div(x)
        x = Rearrange('b c h w -> b h w c')(x)
        x = self.ln(x)
        x = Rearrange('b h w c -> b c h w')(x)
        return x


class PatchReconstruct(nn.Module):
    def __init__(self, dim, patch_size=[4,4]):
        """ Video tube embedding block
        """
        p_h, p_w = patch_size
        super(PatchReconstruct, self).__init__()
        self.to_patches = Rearrange("b h w (c p_h p_w) -> b c (h p_h) (w p_w)", p_h=p_h, p_w=p_w)
        out_dim = p_h * p_w * 3
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim, bias=False)
        
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.proj(self.ln(x))
        x = self.to_patches(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        """ Patch merging block
        """
        super(PatchMerging, self).__init__()
        self.ln = nn.LayerNorm(in_dim * 4)
        self.proj = nn.Linear(in_dim * 4, out_dim, bias=False)
    
    def forward(self, x):
        # division
        x = Rearrange('b c h w -> b h w c')(x)
        H, W, _ = x.shape[-3:]
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C

        x = self.proj(self.ln(x))  # ... H/2 W/2 2*C
        x = Rearrange('b h w c -> b c h w')(x)
        return x
    

class PatchReverse(nn.Module):
    def __init__(self, in_dim, out_dim):
        """ Patch reverse merging block
        """
        super(PatchReverse, self).__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim*4, bias=False)
    
    def forward(self, x):
        x = Rearrange('b c h w -> b h w c')(x)
        x = self.proj(self.ln(x)) 
        x = Rearrange("b h w (p_h p_w c) -> b c (h p_h) (w p_w)", p_h=2, p_w=2)(x) 
        return x