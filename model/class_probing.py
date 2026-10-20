import torch
import torch.nn as nn

from model.deep_JSCC import SwinJSCC, SwinJSCC_CNN
from model.layers import *

__all__ = ['SwinJSCC_probe', 'SwinJSCC_probe_CNN']

class SwinJSCC_probe(SwinJSCC):
    def __init__(self, exp, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        from utils import get_ckpt
        exp_ckpt = get_ckpt(exp)
        pretrained_states = torch.load(exp_ckpt, map_location="cpu")["state_dict"]

        # load pretrained encoder
        def submodule_state_dict(state_dict, module_name):
            prefix = module_name + "."
            return {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        self.backbones = [self.enc_1, self.enc_2, self.enc_3_0, self.enc_3_1,
                  self.enc_3_2, self.enc_4, self.enc_proj]
        self.enc_1.load_state_dict(submodule_state_dict(pretrained_states, "enc_1"))
        self.enc_2.load_state_dict(submodule_state_dict(pretrained_states, "enc_2"))
        self.enc_3_0.load_state_dict(submodule_state_dict(pretrained_states, "enc_3_0"))
        self.enc_3_1.load_state_dict(submodule_state_dict(pretrained_states, "enc_3_1"))
        self.enc_3_2.load_state_dict(submodule_state_dict(pretrained_states, "enc_3_2"))
        self.enc_4.load_state_dict(submodule_state_dict(pretrained_states, "enc_4"))
        self.enc_proj.load_state_dict(submodule_state_dict(pretrained_states, "enc_proj"))
        
        # freeze backbone
        for m in self.backbones:
            for p in m.parameters():
                p.requires_grad = False
        
        dims = self._get_probe_dims(kwargs)
        def make_probe(in_dim):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),   # [B, C, 1, 1] -> [B, C]
                nn.Linear(in_dim, num_classes)
            )
        self.enc_1_probe = make_probe(dims[0])
        self.enc_2_probe = make_probe(dims[1])
        self.enc_3_0_probe = make_probe(dims[2])
        self.enc_3_1_probe = make_probe(dims[2])
        self.enc_3_2_probe = make_probe(dims[2])
        self.enc_4_probe = make_probe(dims[3])
        self.enc_proj_probe = make_probe(self.symbol_dim)

    @staticmethod
    def _get_probe_dims(kwargs):
        return kwargs["dims"]
        
    def forward(self, x):
        with torch.no_grad():
            x_1 = self.enc_1(x)
            x_2 = self.enc_2(x_1)
            x_3_0 = self.enc_3_0(x_2)
            x_3_1 = self.enc_3_1(x_3_0)
            x_3_2 = self.enc_3_2(x_3_1)
            x_4 = self.enc_4(x_3_2)
            y = self.enc_proj(x_4)
        
        # probe logits
        l_1 = self.enc_1_probe(x_1)
        l_2 = self.enc_2_probe(x_2)
        l_3_0 = self.enc_3_0_probe(x_3_0)
        l_3_1 = self.enc_3_1_probe(x_3_1)
        l_3_2 = self.enc_3_2_probe(x_3_2)
        l_4 = self.enc_4_probe(x_4)
        l_y = self.enc_proj_probe(y)

        return {
            "l_1": l_1, "l_2": l_2, "l_3_0": l_3_0, "l_3_1": l_3_1, "l_3_2": l_3_2, "l_4": l_4, "l_y": l_y, 
        }

    def train(self, mode: bool = True):
        super().train(mode)

        # frozen backbone은 항상 eval 유지
        for m in self.backbones:
            m.eval()

        # probe만 mode 따라감
        self.enc_1_probe.train(mode)
        self.enc_2_probe.train(mode)
        self.enc_3_0_probe.train(mode)
        self.enc_3_1_probe.train(mode)
        self.enc_3_2_probe.train(mode)
        self.enc_4_probe.train(mode)
        self.enc_proj_probe.train(mode)

        return self

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.exp', type=str)
        parser.add_argument('--model.num_classes', type=int)
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dims', type=tuple)
        parser.add_argument('--model.img_shape', type=tuple)
        parser.add_argument('--model.patch_shape', type=tuple)
        parser.add_argument('--model.window_size', type=int) 
        parser.add_argument('--model.head_dim', type=int) 
        
        return parser
    
    def state_dict(self, *args, **kwargs):
        original_state_dict = super().state_dict(*args, **kwargs)
        probe_state_dict = {
            k: v for k, v in original_state_dict.items()
            if "probe" in k
        }
        return probe_state_dict
    
    def load_state_dict(self, state_dict, strict=False, assign=False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class SwinJSCC_probe_CNN(SwinJSCC_probe, SwinJSCC_CNN):
    """Linear probes over a frozen :class:`SwinJSCC_CNN` encoder."""

    def __init__(self, exp, num_classes=10, dim=384, **kwargs):
        super().__init__(
            exp=exp,
            num_classes=num_classes,
            dim=dim,
            **kwargs,
        )

    @staticmethod
    def _get_probe_dims(kwargs):
        dim = kwargs["dim"]
        return dim, dim, 2 * dim, 2 * dim

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.exp', type=str)
        parser.add_argument('--model.num_classes', type=int)
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        return parser
