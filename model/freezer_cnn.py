import torch

from model.channel import PseudoRicianChannel
from model.deep_JSCC import SwinJSCC_CNN
from model.split_cnn import SwinJSCC_split_CNN

__all__ = [
    'SwinJSCC_slideEnc_ftDec_CNN',
    'SwinJSCC_frzEnc_ftDec_CNN',
    'SwinJSCC_frzEnc_ftDec_Rician_CNN',
]


class SwinJSCC_slideEnc_ftDec_CNN(SwinJSCC_split_CNN):
    """Fine-tune a sliding encoder window and the full CNN decoder."""

    def __init__(self, exp, start_idx=0, num_train_blocks=4, **kwargs):
        super().__init__(**kwargs)

        assert start_idx >= 0, f"start_idx must be >= 0, got {start_idx}"
        assert num_train_blocks >= 1, \
            f"num_train_blocks must be >= 1, got {num_train_blocks}"
        assert start_idx + num_train_blocks <= self.NUM_BLOCKS, \
            f"start_idx + num_train_blocks must be <= {self.NUM_BLOCKS}, " \
            f"got {start_idx} + {num_train_blocks} = " \
            f"{start_idx + num_train_blocks}"

        self.start_idx = start_idx
        self.num_train_blocks = num_train_blocks

        from utils import get_ckpt
        exp_ckpt = get_ckpt(exp)
        pretrained_states = torch.load(
            exp_ckpt, map_location="cpu"
        )["state_dict"]
        self.load_state_dict(pretrained_states, strict=True)

        self.trainable_enc_idx = set(
            range(start_idx, start_idx + num_train_blocks)
        )
        self.trainable_modules = []
        self.frozen_modules = []

        for idx, block in enumerate(self.enc_blocks):
            if idx in self.trainable_enc_idx:
                self.trainable_modules.append(block)
            else:
                self.frozen_modules.append(block)
        self.trainable_modules.extend(self.dec_blocks)

        for module in self.trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
            module.train()

        for module in self.frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        for module in self.frozen_modules:
            module.eval()
        for module in self.trainable_modules:
            module.train(mode)

        return self

    def state_dict(self, *args, **kwargs):
        full_state = super().state_dict(*args, **kwargs)

        def is_trainable(key):
            if key.startswith("dec_blocks."):
                return True
            if key.startswith("enc_blocks."):
                block_idx = int(key.split(".")[1])
                return block_idx in self.trainable_enc_idx
            return False

        return {
            key: value
            for key, value in full_state.items()
            if is_trainable(key)
        }

    def load_state_dict(self, state_dict, strict=False, assign=False):
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.exp', type=str)
        parser.add_argument('--model.start_idx', type=int, default=0)
        parser.add_argument('--model.num_train_blocks', type=int, default=4)
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        parser.add_argument('--model.channel_type', type=str)
        parser.add_argument('--model.k_factor', type=float)
        return parser


class SwinJSCC_frzEnc_ftDec_CNN(SwinJSCC_CNN):
    """Fine-tune part of the CNN encoder and the full decoder."""

    def __init__(self, exp, num_tune=3, tune_outside=False, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= num_tune <= 7, \
            f"num_tune must be in [0, 7], got {num_tune}"

        from utils import get_ckpt
        exp_ckpt = get_ckpt(exp)
        pretrained_states = torch.load(
            exp_ckpt, map_location="cpu"
        )["state_dict"]

        def submodule_state_dict(state_dict, module_name):
            prefix = module_name + "."
            out = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if len(out) == 0:
                raise KeyError(f"No parameters found for '{module_name}'")
            return out

        self.encoder_modules = [
            ("enc_1", self.enc_1),
            ("enc_2", self.enc_2),
            ("enc_3_0", self.enc_3_0),
            ("enc_3_1", self.enc_3_1),
            ("enc_3_2", self.enc_3_2),
            ("enc_4", self.enc_4),
            ("enc_proj", self.enc_proj),
        ]
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

        for name, module in self.all_ft_modules:
            module.load_state_dict(
                submodule_state_dict(pretrained_states, name)
            )

        if num_tune == 0:
            trainable_encoder = []
        elif tune_outside:
            trainable_encoder = self.encoder_modules[:num_tune]
        else:
            trainable_encoder = self.encoder_modules[-num_tune:]

        self.trainable_module_names = {
            name
            for name, _ in trainable_encoder + self.decoder_modules
        }
        self.trainable_modules = [
            module
            for name, module in self.all_ft_modules
            if name in self.trainable_module_names
        ]
        self.frozen_modules = [
            module
            for name, module in self.all_ft_modules
            if name not in self.trainable_module_names
        ]

        for name, module in self.all_ft_modules:
            is_trainable = name in self.trainable_module_names
            for parameter in module.parameters():
                parameter.requires_grad = is_trainable
            if is_trainable:
                module.train()
            else:
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        for module in self.frozen_modules:
            module.eval()
        for module in self.trainable_modules:
            module.train(mode)

        return self

    def state_dict(self, *args, **kwargs):
        full_state = super().state_dict(*args, **kwargs)
        trainable_prefixes = tuple(
            name + "." for name in self.trainable_module_names
        )
        return {
            key: value
            for key, value in full_state.items()
            if key.startswith(trainable_prefixes)
        }

    def load_state_dict(self, state_dict, strict=False, assign=False):
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )

    @staticmethod
    def get_parser(parser):
        parser.add_argument('--model.exp', type=str)
        parser.add_argument('--model.num_tune', type=int, default=3)
        parser.add_argument('--model.tune_outside', action='store_true')
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple)
        return parser


class SwinJSCC_frzEnc_ftDec_Rician_CNN(SwinJSCC_frzEnc_ftDec_CNN):
    def __init__(self, k_factor, SNRdB=10, **kwargs):
        super().__init__(SNRdB=SNRdB, **kwargs)
        self.channel = PseudoRicianChannel(K=k_factor, SNRdB=SNRdB)

    @staticmethod
    def get_parser(parser):
        parser = SwinJSCC_frzEnc_ftDec_CNN.get_parser(parser)
        parser.add_argument('--model.k_factor', type=float)
        return parser
