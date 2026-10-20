import torch
import torch.nn as nn

from einops import rearrange
from model.deep_JSCC import SwinJSCC
from model.layers import *
from model.channel import PseudoRicianChannel

__all__ = [
    'SwinJSCC_freeze',
    'SwinJSCC_freeze_enc',
    'SwinJSCC_freeze_enc_Rician',
]

class SwinJSCC_freeze(SwinJSCC):
    def __init__(self, exp, num_tune=3, tune_outside=False, **kwargs):
        super().__init__(**kwargs)
        assert 1 <= num_tune <= 7, f"num_tune must be in [1, 7], got {num_tune}"

        from utils import get_ckpt
        exp_ckpt = get_ckpt(exp)
        pretrained_states = torch.load(exp_ckpt, map_location="cpu")["state_dict"]

        def submodule_state_dict(state_dict, module_name):
            prefix = module_name + "."
            out = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if len(out) == 0:
                raise KeyError(f"No parameters found for '{module_name}'")
            return out

        # encoder: outside -> inside
        self.encoder_modules = [
            ("enc_1", self.enc_1),
            ("enc_2", self.enc_2),
            ("enc_3_0", self.enc_3_0),
            ("enc_3_1", self.enc_3_1),
            ("enc_3_2", self.enc_3_2),
            ("enc_4", self.enc_4),
            ("enc_proj", self.enc_proj),
        ]

        # decoder: inside -> outside
        self.decoder_modules = [
            ("dec_proj", self.dec_proj),
            ("dec_1", self.dec_1),
            ("dec_2_0", self.dec_2_0),
            ("dec_2_1", self.dec_2_1),
            ("dec_2_2", self.dec_2_2),
            ("dec_3", self.dec_3),
            ("dec_4", self.dec_4),
        ]

        self.all_ft_modules = self.encoder_modules + self.decoder_modules

        # load full pretrained
        for name, module in self.all_ft_modules:
            module.load_state_dict(submodule_state_dict(pretrained_states, name))

        # choose trainable modules
        if not tune_outside:
            trainable_encoder = self.encoder_modules[-num_tune:]
            trainable_decoder = self.decoder_modules[:num_tune]
        else:
            trainable_encoder = self.encoder_modules[:num_tune]
            trainable_decoder = self.decoder_modules[-num_tune:]

        self.trainable_module_names = set(
            [name for name, _ in trainable_encoder + trainable_decoder]
        )
        self.trainable_modules = [
            module for name, module in self.all_ft_modules
            if name in self.trainable_module_names
        ]
        self.frozen_modules = [
            module for name, module in self.all_ft_modules
            if name not in self.trainable_module_names
        ]

        # freeze / unfreeze
        for name, module in self.all_ft_modules:
            is_trainable = name in self.trainable_module_names
            for p in module.parameters():
                p.requires_grad = is_trainable
            if is_trainable:
                module.train()
            else:
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        for m in self.frozen_modules:
            m.eval()

        for m in self.trainable_modules:
            m.train(mode)

        return self

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.exp', type=str)
        parser.add_argument('--model.num_tune', type=int, default=3)
        parser.add_argument('--model.tune_outside', action='store_true')
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dims', type=tuple)
        parser.add_argument('--model.img_shape', type=tuple)
        parser.add_argument('--model.patch_shape', type=tuple)
        parser.add_argument('--model.window_size', type=int)
        parser.add_argument('--model.head_dim', type=int)
        return parser

    def state_dict(self, *args, **kwargs):
        full_state = super().state_dict(*args, **kwargs)
        trainable_prefixes = tuple(name + "." for name in self.trainable_module_names)
        return {
            k: v for k, v in full_state.items()
            if k.startswith(trainable_prefixes)
        }

    def load_state_dict(self, state_dict, strict=False, assign=False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class SwinJSCC_freeze_enc(SwinJSCC_freeze):
    def __init__(self, exp, num_tune=3, tune_outside=False, **kwargs):
        SwinJSCC.__init__(self, **kwargs)
        assert 1 <= num_tune <= 7, f"num_tune must be in [1, 7], got {num_tune}"

        from utils import get_ckpt
        exp_ckpt = get_ckpt(exp)
        pretrained_states = torch.load(exp_ckpt, map_location="cpu")["state_dict"]

        def submodule_state_dict(state_dict, module_name):
            prefix = module_name + "."
            out = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if len(out) == 0:
                raise KeyError(f"No parameters found for '{module_name}'")
            return out

        # encoder: outside -> inside
        self.encoder_modules = [
            ("enc_1", self.enc_1),
            ("enc_2", self.enc_2),
            ("enc_3_0", self.enc_3_0),
            ("enc_3_1", self.enc_3_1),
            ("enc_3_2", self.enc_3_2),
            ("enc_4", self.enc_4),
            ("enc_proj", self.enc_proj),
        ]

        # decoder: inside -> outside
        self.decoder_modules = [
            ("dec_proj", self.dec_proj),
            ("dec_1", self.dec_1),
            ("dec_2_0", self.dec_2_0),
            ("dec_2_1", self.dec_2_1),
            ("dec_2_2", self.dec_2_2),
            ("dec_3", self.dec_3),
            ("dec_4", self.dec_4),
        ]

        self.all_ft_modules = self.encoder_modules + self.decoder_modules

        # load full pretrained
        for name, module in self.all_ft_modules:
            module.load_state_dict(submodule_state_dict(pretrained_states, name))

        # choose trainable modules
        if not tune_outside:
            trainable_encoder = self.encoder_modules[-num_tune:]
        else:
            trainable_encoder = self.encoder_modules[:num_tune]

        self.trainable_module_names = set(
            [name for name, _ in trainable_encoder]
        )
        self.trainable_modules = [
            module for name, module in self.all_ft_modules
            if name in self.trainable_module_names
        ]
        self.frozen_modules = [
            module for name, module in self.all_ft_modules
            if name not in self.trainable_module_names
        ]

        # freeze / unfreeze
        for name, module in self.all_ft_modules:
            is_trainable = name in self.trainable_module_names
            for p in module.parameters():
                p.requires_grad = is_trainable
            if is_trainable:
                module.train()
            else:
                module.eval()


class SwinJSCC_freeze_enc_Rician(SwinJSCC_freeze_enc):
    def __init__(self, k_factor, SNRdB=10, **kwargs):
        super().__init__(SNRdB=SNRdB, **kwargs)
        self.channel = PseudoRicianChannel(K=k_factor, SNRdB=SNRdB)

    @staticmethod
    def get_parser(parser):
        parser = SwinJSCC_freeze_enc.get_parser(parser)
        parser.add_argument('--model.k_factor', type=float)
        return parser
