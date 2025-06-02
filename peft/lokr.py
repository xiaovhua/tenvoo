
import math
from dataclasses import dataclass, field
import monai

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_config import PeftType, PeftLayer, PeftModel, MergeBuffer
from .base_config import PeftConvConfig as PeftConfig


@dataclass
class LokrConfig(PeftConfig):
    merge_weights: bool = field(default=True, metadata={"help": "merge weights of the original model and the lora model"})
    alpha: float = field(default = 0.0, metadata={"help": "weight of lora matrix"})
    factor: int = field(default = -1, metadata={"help": "dimension of lokr factor"})
    def __post_init__(self):
        self.peft_type = PeftType.LOKR


class LokrModel(PeftModel):
    def __init__(self, config, model):
        super().__init__(config, model)            
    def get_linear_args(self, target):
        kwargs = {
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode),
            "rank": self.peft_config.rank,
            "alpha": self.peft_config.alpha,
            "factor": self.peft_config.factor
        }
        return kwargs

    def get_conv_args(self, target):
        kwargs = {
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode),
            "rank": self.peft_config.rank,
            "alpha": self.peft_config.alpha,
            "factor": self.peft_config.factor
        }
        return kwargs

    def get_new_linear_module(self, in_features, out_features, bias=True, **kwargs):
        return Linear(in_features, out_features, bias=bias, **kwargs)

    def get_new_conv_module(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1, groups=1, **kwargs):
        return Conv3d(in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, **kwargs)


class LokrLayer(PeftLayer):
    def __init__(self, merge_weights: bool):
        self.merged = MergeBuffer(default=False)  # so that this will be tracked when saving the model and loading it
        self.merge_weights = merge_weights


# from https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/functional/general.py
def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    second value is a value for weight.

    Because of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1, w2, scale):
    for _ in range(w2.dim() - w1.dim()):
        w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


class Linear(nn.Linear, LokrLayer):
    # Lokr implemented in a dense layer
    def __init__(self, in_features: int, out_features: int, bias: bool,  merge_weights: bool = False, rank: int = 16, alpha: float = 0., factor: int = -1, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        nn.Linear.reset_parameters(self)

        LokrLayer.__init__(self, merge_weights=merge_weights)

        # layer
        in_m, in_n = factorization(in_features, factor)
        out_l, out_k = factorization(out_features, factor)
        shape = (
            (out_l, out_k),
            (in_m, in_n),
        )
        self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], rank))
        self.lokr_w1_b = nn.Parameter(torch.empty(rank, shape[1][0]))
        self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], rank))
        self.lokr_w2_b = nn.Parameter(torch.empty(rank, shape[1][1]))

        # scale
        alpha = rank // 4 if alpha is None or alpha == 0 else alpha
        self.scale = alpha / rank

        # initialize
        torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
        torch.nn.init.constant_(self.lokr_w2_b, 0)

    def gen_full_weights(self):
        w1 = self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2_a @ self.lokr_w2_b
        # kron product
        full_weights = make_kron(w1, w2, self.scale)
        # print('** Linear: ', full_weights.shape, self.weight.shape)
        return full_weights.view(self.weight.shape)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                full_weights = self.gen_full_weights()
                self.weight.data -= full_weights
                self.merged.set(False)
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                full_weights = self.gen_full_weights()
                self.weight.data += full_weights
                self.merged.set(True)

    def forward(self, x: torch.Tensor):
        if isinstance(x, monai.data.meta_tensor.MetaTensor):
            x = x.as_tensor()
        previous_dtype = self.weight.dtype
        if not self.merged:
            return F.linear(x, self.weight + self.gen_full_weights(), bias=self.bias.to(previous_dtype) if self.bias is not None else None)
        else:
            return F.linear(x, self.weight, bias=self.bias.to(previous_dtype) if self.bias is not None else None)


class Conv3d(nn.Conv3d, LokrLayer):
    # Lokr implemented in a convolutional layer
    def __init__(self, in_channels: int, out_channels: int, bias: bool, kernel_size: int | list | tuple = 1,
                 stride: int | list | tuple = 1, padding: int | list | tuple = 1, dilation: int | tuple = 1,
                 # groups: int = 1, 
                 merge_weights: bool = False, rank: int = 4, alpha: float = 0.0, factor: int = -1, **kwargs):

        # nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, **kwargs)
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, **kwargs)
        nn.Conv3d.reset_parameters(self)

        LokrLayer.__init__(self, merge_weights=merge_weights)
        
        # layer
        in_m, in_n = factorization(in_channels, factor)
        out_l, out_k = factorization(out_channels, factor)
        shape = (
            (out_l, out_k),
            (in_m, in_n),
            *self.kernel_size
        )
        self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], rank))
        self.lokr_w1_b = nn.Parameter(torch.empty(rank, shape[1][0]))
        self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], rank))
        self.lokr_w2_b = nn.Parameter(torch.empty(rank, shape[1][1] * torch.tensor(shape[2:]).prod().item()))
    
        # scale
        alpha = rank // 4 if alpha is None or alpha == 0 else alpha
        self.scale = alpha / rank

        # initialize
        torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
        torch.nn.init.constant_(self.lokr_w2_b, 0)

    
    def gen_full_weights(self):
        w1 = self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2_a @ self.lokr_w2_b
        # kron product
        full_weights = make_kron(w1, w2, self.scale)
        # print('++ Conv: ', full_weights.shape, self.weight.shape)
        return full_weights.view(self.weight.shape)


    def train(self, mode: bool = True):
        nn.Conv3d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                self.weight.data -= self.gen_full_weights()
                self.merged.set(False)
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data += self.gen_full_weights()
                self.merged.set(True)


    def forward(self, x: torch.Tensor):
        if isinstance(x, monai.data.meta_tensor.MetaTensor):
            x = x.as_tensor()
        previous_dtype = self.weight.dtype
        if not self.merged:
            return F.conv3d(x, self.weight + self.gen_full_weights(), stride=self.stride, padding=self.padding, dilation=self.dilation, 
                    groups=self.groups,  bias=self.bias.to(previous_dtype) if self.bias is not None else None)
        else:
            return F.conv3d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, 
                    bias=self.bias.to(previous_dtype) if self.bias is not None else None)


