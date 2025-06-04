from dataclasses import dataclass, field
import monai
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_config import PeftType, PeftLayer, PeftModel, MergeBuffer
from .base_config import PeftConvConfig as PeftConfig


@dataclass
class LohaConfig(PeftConfig):
    merge_weights: bool = field(default=True, metadata={"help": "merge weights of the original model and the lora model"})
    alpha: float = field(default = 0.0, metadata={"help": "weight of lora matrix"})
    def __post_init__(self):
        self.peft_type = PeftType.LOHA


class LohaModel(PeftModel):
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

    def get_new_conv_module(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1, groups=1, **kwargs):
        return Conv3d(in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, **kwargs)



class LohaLayer(PeftLayer):
    def __init__(self, merge_weights: bool):
        self.merged = MergeBuffer(default=False)  # so that this will be tracked when saving the model and loading it
        self.merge_weights = merge_weights


# from https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/functional/loha.py
class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1d, w1u, w2d, w2u, scale=torch.tensor(1)):
        ctx.save_for_backward(w1d, w1u, w2d, w2u, scale)
        diff_weight = ((w1u @ w1d) * (w2u @ w2d)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1d, w1u, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2u @ w2d)
        grad_w1u = temp @ w1d.T
        grad_w1d = w1u.T @ temp

        temp = grad_out * (w1u @ w1d)
        grad_w2u = temp @ w2d.T
        grad_w2d = w2u.T @ temp

        del temp
        return grad_w1d, grad_w1u, grad_w2d, grad_w2u, None


class HadaWeightTucker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1d, w1u, t2, w2d, w2u, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1d, w1u, t2, w2d, w2u, scale)

        rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", t1, w1d, w1u)
        rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", t2, w2d, w2u)

        return rebuild1 * rebuild2 * scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1d, w1u, t2, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale

        temp = torch.einsum("i j ..., j r -> i r ...", t2, w2d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w2u)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w1u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w1u.T)
        del grad_w, temp

        grad_w1d = torch.einsum("i r ..., i j ... -> r j", t1, grad_temp)
        grad_t1 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w1d.T)
        del grad_temp

        temp = torch.einsum("i j ..., j r -> i r ...", t1, w1d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w1u)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w2u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w2u.T)
        del grad_w, temp

        grad_w2d = torch.einsum("i r ..., i j ... -> r j", t2, grad_temp)
        grad_t2 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w2d.T)
        del grad_temp
        return grad_t1, grad_w1d, grad_w1u, grad_t2, grad_w2d, grad_w2u, None


def make_weight(w1d, w1u, w2d, w2u, scale):
    return HadaWeight.apply(w1d, w1u, w2d, w2u, scale)


def make_weight_tucker(t1, w1d, w1u, t2, w2d, w2u, scale):
    return HadaWeightTucker.apply(t1, w1d, w1u, t2, w2d, w2u, scale)


def weight_gen(org_weight, rank, tucker=True):
    """### weight_gen

    Args:
        org_weight (torch.Tensor): the weight tensor
        rank (int): low rank

    Returns:
        torch.Tensor: w1d, w2d, w1u, w2u[, t1, t2]
    """
    out_dim, in_dim, *k = org_weight.shape
    if k and tucker:
        w1d = torch.empty(rank, in_dim)
        w1u = torch.empty(rank, out_dim)
        t1 = torch.empty(rank, rank, *k)
        w2d = torch.empty(rank, in_dim)
        w2u = torch.empty(rank, out_dim)
        t2 = torch.empty(rank, rank, *k)
        nn.init.normal_(t1, std=0.1)
        nn.init.normal_(t2, std=0.1)
    else:
        w1d = torch.empty(rank, in_dim)
        w1u = torch.empty(out_dim, rank)
        w2d = torch.empty(rank, in_dim)
        w2u = torch.empty(out_dim, rank)
        t1 = t2 = None
    nn.init.normal_(w1d, std=1)
    nn.init.constant_(w1u, 0)
    nn.init.normal_(w2d, std=1)
    nn.init.normal_(w2u, std=0.1)
    return w1d, w1u, w2d, w2u, t1, t2


def diff_weight(*weights, gamma=1.0):
    """### diff_weight

    Get ΔW = BA, where BA is low rank decomposition

    Args:
        wegihts (tuple[torch.Tensor]): (w1d, w2d, w1u, w2u[, t1, t2])
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: ΔW
    """
    w1d, w1u, w2d, w2u, t1, t2 = weights
    if t1 is not None and t2 is not None:
        R, I = w1d.shape
        R, O = w1u.shape
        R, R, *k = t1.shape
        result = make_weight_tucker(t1, w1d, w1u, t2, w2d, w2u, gamma)
    else:
        R, I, *k = w1d.shape
        O, R, *_ = w1u.shape
        w1d = w1d.reshape(w1d.size(0), -1)
        w1u = w1u.reshape(-1, w1u.size(1))
        w2d = w2d.reshape(w2d.size(0), -1)
        w2u = w2u.reshape(-1, w2u.size(1))
        result = make_weight(w1d, w1u, w2d, w2u, gamma)

    result = result.reshape(O, I, *k)
    return result


class Linear(LohaLayer, nn.Linear):
    # Loha implemented in a dense layer
    def __init__(self, in_features: int, out_features: int, bias: bool,  merge_weights: bool = False, rank: int = 16, alpha: float = 0., **kwargs):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        nn.Linear.reset_parameters(self)

        LohaLayer.__init__(self, merge_weights=merge_weights)

        # layer
        self.loha_w1_a = nn.Parameter(torch.empty(in_features, rank))
        self.loha_w1_b = nn.Parameter(torch.empty(rank, out_features))
        self.loha_w2_a = nn.Parameter(torch.empty(in_features, rank))
        self.loha_w2_b = nn.Parameter(torch.empty(rank, out_features))

        # scale
        alpha = rank // 4 if alpha is None or alpha == 0 else alpha
        self.scale = alpha / rank

        # initialize
        torch.nn.init.normal_(self.loha_w1_a, std=0.1)
        torch.nn.init.normal_(self.loha_w1_b, std=1.)
        torch.nn.init.constant_(self.loha_w2_a, 0)
        torch.nn.init.normal_(self.loha_w2_b, std=1.)

    def gen_full_weights(self):
        scale = torch.tensor(
            self.scale, dtype=self.loha_w1_b.dtype, device=self.loha_w1_b.device
        )
        # hadamard product
        full_weights = diff_weight(
            self.loha_w1_b,
            self.loha_w1_a,
            self.loha_w2_b,
            self.loha_w2_a,
            None,
            None,
            gamma=scale
        )
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


class Conv3d(LohaLayer, nn.Conv3d):
    # Loha implemented in a convolutional layer
    def __init__(self, in_channels: int, out_channels: int, bias: bool, kernel_size: int | list | tuple = 1,
                 stride: int | list | tuple = 1, padding: int | list | tuple = 1, dilation: int | tuple = 1,
                 # groups: int = 1, 
                 merge_weights: bool = False, rank: int = 4, alpha: float = 0.0, **kwargs):

        # nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, **kwargs)
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, **kwargs)
        nn.Conv3d.reset_parameters(self)

        LohaLayer.__init__(self, merge_weights=merge_weights)
        
        # layer
        self.loha_w1_a = nn.Parameter(torch.empty(out_channels, rank))
        self.loha_w1_b = nn.Parameter(torch.empty(rank, in_channels * torch.tensor(self.kernel_size).prod().item()))
        self.loha_w2_a = nn.Parameter(torch.empty(out_channels, rank))
        self.loha_w2_b = nn.Parameter(torch.empty(rank, in_channels * torch.tensor(self.kernel_size).prod().item()))
    
        # scale
        alpha = rank // 4 if alpha is None or alpha == 0 else alpha
        self.scale = alpha / rank

        # initialize
        torch.nn.init.normal_(self.loha_w1_a, std=0.1)
        torch.nn.init.normal_(self.loha_w1_b, std=1.)
        torch.nn.init.constant_(self.loha_w2_a, 0)
        torch.nn.init.normal_(self.loha_w2_b, std=1.)

    
    def gen_full_weights(self):
        scale = torch.tensor(
            self.scale, dtype=self.loha_w1_b.dtype, device=self.loha_w1_b.device
        )
        full_weights = diff_weight(
            self.loha_w1_b,
            self.loha_w1_a,
            self.loha_w2_b,
            self.loha_w2_a,
            None,
            None,
            gamma=scale
        )
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

