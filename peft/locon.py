import math
from dataclasses import dataclass, field
import monai
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_config import PeftType, PeftLayer, PeftModel, MergeBuffer
from .base_config import PeftConvConfig as PeftConfig


@dataclass
class LoConConfig(PeftConfig):
    merge_weights: bool = field(default=True, metadata={"help": "merge weights of the original model and the lora model"})
    alpha: float = field(default = 0.0, metadata={"help": "weight of lora matrix"})

    def __post_init__(self):
        self.peft_type = PeftType.LOCON


class LoConModel(PeftModel):
    def __init__(self, config, model):
        super().__init__(config, model)
                
    def get_linear_args(self, target):
        kwargs = {
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode),
            "rank": self.peft_config.rank,
            "alpha": self.peft_config.alpha
        }
        return kwargs

    def get_conv_args(self, target):
        kwargs = {
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode),
            "rank": self.peft_config.rank,
            "alpha": self.peft_config.alpha
        }
        return kwargs

    def get_new_linear_module(self, in_features, out_features, bias=True, **kwargs):
        return Linear(in_features, out_features, bias=bias, **kwargs)

    def get_new_conv_module(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1, **kwargs):
        return Conv3d(in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)


class LoConLayer(PeftLayer):
    def __init__(self, merge_weights: bool):
        self.merged = MergeBuffer(default=False)  # so that this will be tracked when saving the model and loading it
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoConLayer):
    # LoCon implemented in a dense layer
    def __init__(self, in_features: int, out_features: int, bias: bool,  merge_weights: bool = False, rank: int = 16, alpha: float = 0., **kwargs):

        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        nn.Linear.reset_parameters(self)

        LoConLayer.__init__(self, merge_weights=merge_weights)

        # layer
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # scale
        alpha = rank // 4 if alpha is None or alpha == 0 else alpha
        self.scale = alpha / rank

        # initialize
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.constant_(self.lora_up.weight, 0)

    def gen_full_weights(self):
        wa = self.lora_up.weight
        wb = self.lora_down.weight
        full_weights = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
        # print('** Linear **:', full_weights.shape, self.weight.shape)
        return self.scale * full_weights.view(self.weight.shape)

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


class Conv3d(nn.Conv3d, LoConLayer):
    # LoCon implemented in a convolutional layer
    def __init__(self, in_channels: int, out_channels: int, bias: bool, kernel_size: int | list | tuple = 1,
                 stride: int | list | tuple = 1, padding: int | list | tuple = 1, dilation: int | tuple = 1,
                 # groups: int = 1, 
                 merge_weights: bool = False, rank: int = 4, alpha: float = 0.0,  **kwargs):

        # nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, **kwargs)
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, **kwargs)
        nn.Conv3d.reset_parameters(self)

        LoConLayer.__init__(self, merge_weights=merge_weights)
        
        # layer
        self.lora_down = nn.Conv3d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.lora_up = nn.Conv3d(rank, out_channels, 1, bias=False)
    
        # scale
        alpha = rank // 4 if alpha is None or alpha == 0 else alpha
        self.scale = alpha / rank

        # initialize
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.constant_(self.lora_up.weight, 0)


    def gen_full_weights(self):
        wa = self.lora_up.weight
        wb = self.lora_down.weight
        full_weights = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
        return self.scale * full_weights.view(self.weight.shape)


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



