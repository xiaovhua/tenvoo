import importlib
import itertools
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import prod
from typing import List, Optional, Union, Dict
import enum
import monai

import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_config import PeftType, PeftLayer, PeftModel, mark_lora_layernorm_cls_trainable, MergeBuffer
from .base_config import PeftConvConfig as PeftConfig

TENVOO_LIST = {
    32: [4, 4, 2],
    64: [8, 4, 2],
    128: [8, 8, 2],
    256: [8, 8, 4],
    512: [16, 8, 4],
    1024: [16, 16, 4],
    768: [16, 8, 6],
    384: [8, 8, 6],
    192: [8, 6, 4],
    96: [8, 4, 3],
    1: [1, 1, 1],
    2: [2, 1, 1],
    4: [2, 2, 1],
    8: [2, 2, 2],
    16: [4, 2, 2],
    48: [8, 3, 2],
    24: [4, 3, 2],
    12: [4, 3, 1],
    6: [3, 2, 1]
}


class BufferDict(nn.Module):
    def __init__(self, init_dict=None):
        super(BufferDict, self).__init__()
        self.buffer_names = []
        if init_dict is not None:
            for name, tensor in init_dict.items():
                self.add_buffer(name, tensor)

    def add_buffer(self, name, tensor):
        self.register_buffer(name, tensor)
        if name not in self.buffer_names:
            self.buffer_names.append(name)

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, tensor):
        self.add_buffer(name, tensor)

    def items(self):
        for name in self.buffer_names:
            yield name, getattr(self, name)

    def keys(self):
        return self.buffer_names

    def values(self):
        return [getattr(self, name) for name in self.buffer_names]


@dataclass
class TenVOOConfig(PeftConfig):
    d_in: int = field(default=1, metadata={"help": "quanta number of input dimensions (only for linear)"})
    d_out: int = field(default=1, metadata={"help": "quanta number of output dimensions (only for linear)"})
    per_dim_list: Dict = field(default=None, metadata={"help": "Dictionary of the number features per dimention."})
    dropout: float = field(default=0.0, metadata={"help": "dropout"})
    merge_weights: bool = field(default=True, metadata={"help": "Merge weights of the original model and the Lora model"})
    sum_mode: bool = field(default=False, metadata={"help": "Set this to True if the quanta is in sum mode"})
    initialize_mode: str = field(default='sum_opposite_freeze_one', metadata={"help": "initialization method"})
    model_mode: str = field(default='l', metadata={"help": "type of model, tenvoo-l or tenvoo-q"})

    def __post_init__(self):
        self.peft_type = PeftType.TENVOO


class TenVOOModel(PeftModel):
    def __init__(self, config, model):
        super().__init__(config, model)

    def get_linear_args(self, target):
        kwargs = {
            "d_in": self.peft_config.d_in,
            "d_out": self.peft_config.d_out,
            "quanta_dropout": self.peft_config.dropout,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode),
            "sum_mode": self.peft_config.sum_mode,
            "rank": self.peft_config.rank,
            "per_dim_features": self.peft_config.per_dim_list[target.in_features],
            "per_dim_features2": self.peft_config.per_dim_list[target.out_features],
            "initialize_mode": self.peft_config.initialize_mode
        }
        return kwargs

    def get_conv_args(self, target):
        kwargs = {
            "tenvoo_dropout": self.peft_config.dropout,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode),
            "sum_mode": self.peft_config.sum_mode,
            "rank": self.peft_config.rank,
            "per_dim_features": self.peft_config.per_dim_list[target.in_channels],
            "per_dim_features2": self.peft_config.per_dim_list[target.out_channels],
            "initialize_mode": self.peft_config.initialize_mode,
            "model_mode": self.peft_config.model_mode,
        }

        return kwargs

    def get_new_linear_module(self, in_features, out_features, bias=True, **kwargs):
        return Linear(in_features, out_features, bias=bias, **kwargs)

    def get_new_conv_module(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1, groups=1, **kwargs):
        return Conv3d(in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, **kwargs)


class TenVOOLayer(PeftLayer):
    def __init__(self, d_in: int, d_out: int, dropout: float, merge_weights: bool, sum_mode: bool = False, ):
        self.d_in = d_in
        self.d_out = d_out
        self.sum_mode = sum_mode
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        self.merged = MergeBuffer(default=False)  # so that this will be tracked when saving the model and loading it
        self.frozen_merged = MergeBuffer(default=False)  # the frozen weights are separately tracked
        self.merge_weights = merge_weights


class Linear(nn.Linear, TenVOOLayer):
    # TenVoo implemented in a dense layer, we directly use QuanTA
    def __init__(self, in_features: int, out_features: int, bias: bool,
                 d_in: int = 1, d_out: int = 1,
                 quanta_dropout: float = 0.,
                 fan_in_fan_out: bool = False,
                 per_dim_features: list = None,
                 per_dim_features2: list = None,
                 merge_weights: bool = False, sum_mode: bool = False,
                 rank: int = 4,
                 initialize_mode: str = 'sum_opposite_freeze_one',  # last_layer_zero, sum_opposite_freeze_one,
                 **kwargs):

        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.initialize_mode = initialize_mode

        # Actual trainable parameters
        if per_dim_features is not None:
            d_in = len(per_dim_features)
        if per_dim_features2 is not None:
            d_out = len(per_dim_features2)
        assert d_in == d_out, f"Expected d_in equal to d_out, but got d_in={d_in}, d_out={d_out}"

        TenVooLayer.__init__(
            self, d_in=d_in, d_out=d_out,
            dropout=quanta_dropout, sum_mode=sum_mode,
            merge_weights=merge_weights
        )

        if d_in > 1:
            self.per_dim_features = per_dim_features
            self.per_dim_features2 = self.per_dim_features if per_dim_features2 is None else per_dim_features2
            self.total_features = prod(self.per_dim_features)
            self.total_features2 = prod(self.per_dim_features2)
            if self.total_features != in_features:
                warnings.warn(
                    f'per_dim_features={self.per_dim_features} does not match in_features={in_features}, this should work but may result in downgraded performance or additional cost. Please make sure this is intended.')
            if self.total_features2 != out_features:
                warnings.warn(
                    f'per_dim_features2={self.per_dim_features2} does not match out_features={out_features}, this should work but may result in downgraded performance or additional cost. Please make sure this is intended.')

            # create quanta weights for input dimensions
            if self.per_dim_features == self.per_dim_features2:
                quanta_weights = {}
                for (dim1, dim2) in itertools.combinations(range(-1, -d_in - 1, -1), 2):
                    quanta_weights[f'{dim1} {dim2}'] = nn.Parameter(
                        self.weight.new_zeros(self.per_dim_features2[dim2], self.per_dim_features2[dim1],
                                              self.per_dim_features[dim2], self.per_dim_features[
                                                  dim1]))  # reverse the order because dim1 is closer to the end
                self.quanta_weights = nn.ParameterDict(quanta_weights)

                # create quanta weights for output dimensions
                quanta_weights2 = {}
                for (dim1, dim2) in itertools.combinations(range(-1, -d_out - 1, -1), 2):
                    quanta_weights2[f'{dim1} {dim2}'] = self.weight.new_zeros(
                        self.per_dim_features2[dim2],
                        self.per_dim_features2[dim1],
                        self.per_dim_features[dim2],
                        self.per_dim_features[dim1]
                    )
                self.quanta_weights2 = BufferDict(quanta_weights2)

            else:
                P1 = lambda d1, d2, d3, d4: nn.Parameter(self.weight.new_zeros(d1, d2, d3, d4))
                quanta_weights = {}
                for di, (dim1, dim2) in enumerate(itertools.combinations(range(-1, -d_in - 1, -1), 2)):
                    if self.d_in == 3:
                        quanta_weights[f'{dim1} {dim2}'] = {
                            0: P1(rank, rank, self.per_dim_features[dim2], self.per_dim_features[dim1]),
                            1: P1(rank, self.per_dim_features2[dim1], self.per_dim_features[dim2], rank),
                            2: P1(self.per_dim_features2[dim2], self.per_dim_features2[dim1], rank, rank),
                        }[di]
                    elif self.d_in == 4:
                        quanta_weights[f'{dim1} {dim2}'] = {
                            0: P1(rank, rank, self.per_dim_features[dim2], self.per_dim_features[dim1]),
                            1: P1(rank, rank, self.per_dim_features[dim2], rank),
                            2: P1(rank, self.per_dim_features2[dim1], self.per_dim_features[dim2], rank),
                            3: P1(rank, rank, rank, rank),
                            4: P1(rank, self.per_dim_features2[dim1], rank, rank),
                            5: P1(self.per_dim_features2[dim2], self.per_dim_features2[dim1], rank, rank),
                        }[di]
                    elif self.d_in == 5:
                        quanta_weights[f'{dim1} {dim2}'] = {
                            0: P1(rank, rank, self.per_dim_features[dim1], self.per_dim_features[dim1]),
                            1: P1(rank, rank, self.per_dim_features[dim2], rank),
                            2: P1(rank, rank, self.per_dim_features[dim2], rank),
                            3: P1(rank, self.per_dim_features2[dim1], self.per_dim_features[dim2], rank),
                            4: P1(rank, rank, rank, rank),
                            5: P1(rank, rank, rank, rank),
                            6: P1(rank, self.per_dim_features2[dim1], rank, rank),
                            7: P1(rank, rank, rank, rank),
                            8: P1(rank, self.per_dim_features2[dim1], rank, rank),
                            9: P1(self.per_dim_features2[dim2], self.per_dim_features2[dim1], rank, rank),
                        }[di]
                    elif self.d_in == 6:
                        quanta_weights[f'{dim1} {dim2}'] = {
                            0: P1(rank, rank, self.per_dim_features[dim2], self.per_dim_features[dim1]),
                            1: P1(rank, rank, self.per_dim_features[dim2], rank),
                            2: P1(rank, rank, self.per_dim_features[dim2], rank),
                            3: P1(rank, rank, self.per_dim_features[dim2], rank),
                            4: P1(rank, self.per_dim_features2[dim1], self.per_dim_features[dim2], rank),
                            5: P1(rank, rank, rank, rank),
                            6: P1(rank, rank, rank, rank),
                            7: P1(rank, rank, rank, rank),
                            8: P1(rank, self.per_dim_features2[dim1], rank, rank),
                            9: P1(rank, rank, rank, rank),
                            10: P1(rank, rank, rank, rank),
                            11: P1(rank, self.per_dim_features2[dim1], rank, rank),
                            12: P1(rank, rank, rank, rank),
                            13: P1(rank, self.per_dim_features2[dim1], rank, rank),
                            14: P1(self.per_dim_features2[dim2], self.per_dim_features2[dim1], rank, rank),
                        }[di]
                self.quanta_weights = nn.ParameterDict(quanta_weights)

                quanta_weights2 = {}
                for k, v in quanta_weights.items():
                    quanta_weights2[k] = nn.Parameter(v.clone().detach(), requires_grad=False)
                self.quanta_weights2 = BufferDict(quanta_weights2)

            # freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        # initialize
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        # create einsum op for training and evaluation
        self.gen_einsum_expr_train()
        self.gen_einsum_expr_eval()

    def reset_parameters(self):
        # reset linear
        nn.Linear.reset_parameters(self)

        if hasattr(self, 'quanta_weights'):
            # reset quanta_weights
            for k, v in self.quanta_weights.items():
                nn.init.kaiming_uniform_(v.view(v.shape[0] * v.shape[1], v.shape[2] * v.shape[3]), a=math.sqrt(5), nonlinearity='linear')  # initialize as if it is a matrix
            # set last layer as zero
            if self.initialize_mode == 'last_layer_zero':
                nn.init.zeros_(self.quanta_weights[list(self.quanta_weights.keys())[-1]])
            # set quanta weigths 2 as the frozen backup of quanta weights
            if self.initialize_mode == 'sum_opposite_freeze_one' or self.initialize_mode == 'last_layer_zero': # define a frozon backup
                for k, v in self.quanta_weights2.items():
                    v[:] = self.quanta_weights[k].data
            else:
                raise NotImplementedError

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.d_in > 0:
                    full_quanta_weights = F.pad(self.einsum_expr_eval(
                        *[self.quanta_weights[f'{dim1} {dim2}'].to(self.weight.dtype) for (dim1, dim2) in
                          itertools.combinations(range(-1, -self.d_in - 1, -1), 2)]).reshape(self.total_features2,
                                                                                             self.total_features),
                                                (0, self.in_features - self.total_features, 0,
                                                 self.out_features - self.total_features2),
                                                'constant', 0.)
                    self.weight.data -= T(full_quanta_weights)
                self.merged.set(False)
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.d_in > 0:
                    full_quanta_weights = F.pad(self.einsum_expr_eval(
                        *[self.quanta_weights[f'{dim1} {dim2}'].to(self.weight.dtype) for (dim1, dim2) in
                          itertools.combinations(range(-1, -self.d_in - 1, -1), 2)]).reshape(self.total_features2,
                                                                                             self.total_features),
                                                (0, self.in_features - self.total_features, 0,
                                                 self.out_features - self.total_features2),
                                                'constant', 0.)
                    self.weight.data += T(full_quanta_weights)
                    if not self.frozen_merged:
                        self.merge_frozen_weights()
                self.merged.set(True)

    def merge_frozen_weights(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.frozen_merged:
            warnings.warn('The frozen weights are already merged, ignoring the request to merge the frozen weights')
        else:
            full_quanta_weights = -F.pad(
                self.einsum_expr_eval(
                    *[self.quanta_weights2[f'{dim1} {dim2}'].to(self.weight.dtype) for (dim1, dim2) in \
                      itertools.combinations(range(-1, -self.d_in - 1, -1), 2)]
                ).reshape(self.total_features2, self.total_features),
                (0, self.in_features - self.total_features, 0, self.out_features - self.total_features2), 'constant', 0.
            )

            self.weight.data += T(full_quanta_weights)
            self.frozen_merged.set(True)

    def unmerge_frozen_weights(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if not self.frozen_merged:
            warnings.warn('The frozen weights are already unmerged, ignoring the request to unmerge the frozen weights')
        else:
            full_quanta_weights = -F.pad(self.einsum_expr_eval(
                *[self.quanta_weights2[f'{dim1} {dim2}'].to(self.weight.dtype) for (dim1, dim2) in
                  itertools.combinations(range(-1, -self.d_in - 1, -1), 2)]).reshape(self.total_features2,
                                                                                     self.total_features),
                                         (0, self.in_features - self.total_features, 0,
                                          self.out_features - self.total_features2), 'constant',
                                         0.)
            self.weight.data -= T(full_quanta_weights)
            self.frozen_merged.set(False)

    def gen_einsum_expr_train(self):
        """
        Generate the einsum expression for the tensorized weights during training.
        """
        d = self.d_in  # 4
        current_symbols_inds = list(range(d))

        eq = '...'
        for i in current_symbols_inds:
            eq += oe.get_symbol(i)

        for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
            symbol_ind1 = current_symbols_inds[dim1]
            symbol_ind2 = current_symbols_inds[dim2]
            symbol_ind3 = symbol_ind1 + d
            symbol_ind4 = symbol_ind2 + d
            eq += ',' + oe.get_symbol(symbol_ind4) + oe.get_symbol(symbol_ind3) + oe.get_symbol(
                symbol_ind2) + oe.get_symbol(
                symbol_ind1)  # reverse order because dim1 is toward the end than dim2 and because of matrix multiplication order convention. Note that this is different from the forward function
            current_symbols_inds[dim1] = symbol_ind3
            current_symbols_inds[dim2] = symbol_ind4

        eq += '->...'
        for i in current_symbols_inds:
            eq += oe.get_symbol(i)

        shapes = [(100,) + tuple(self.per_dim_features)]  # may need to change the 100 to some other value
        for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
            shapes.append((self.per_dim_features[dim2], self.per_dim_features[dim1], self.per_dim_features[dim2],
                           self.per_dim_features[dim1]))

        optimize = 'optimal' if d <= 4 else 'branch-all' if d <= 5 else 'branch-2' if d <= 7 else 'auto'
        expr = oe.contract_expression(eq, *shapes, optimize=optimize)
        self.einsum_eq_train = eq
        self.einsum_expr_train = expr

    def gen_einsum_expr_eval(self):
        """
        Generate the einsum expression for the tensorized weights during evaluation.
        """
        d = self.d_in
        current_symbols_inds = list(range(d))
        init_symbols_inds = [i for i in current_symbols_inds]  # copy

        eq = ''

        for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
            symbol_ind1 = current_symbols_inds[dim1]
            symbol_ind2 = current_symbols_inds[dim2]
            symbol_ind3 = symbol_ind1 + d
            symbol_ind4 = symbol_ind2 + d
            eq += ',' + oe.get_symbol(symbol_ind4) + oe.get_symbol(symbol_ind3) + oe.get_symbol(
                symbol_ind2) + oe.get_symbol(symbol_ind1)  # reverse order because dim1 is toward the end than dim2
            current_symbols_inds[dim1] = symbol_ind3
            current_symbols_inds[dim2] = symbol_ind4

        eq += '->'
        for i in current_symbols_inds:
            eq += oe.get_symbol(i)
        for i in init_symbols_inds:
            # note that this is also the reverse order, so it is the usual matrix multiplication order which is (fan_out, fan_in)
            eq += oe.get_symbol(i)
        eq = eq[1:]

        shapes = []
        for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
            shapes.append((self.per_dim_features[dim2], self.per_dim_features[dim1], self.per_dim_features[dim2],
                           self.per_dim_features[dim1]))

        optimize = 'optimal' if d <= 4 else 'branch-all' if d <= 5 else 'branch-2' if d <= 7 else 'auto'
        expr = oe.contract_expression(eq, *shapes, optimize=optimize)

        self.einsum_eq_eval = eq
        self.einsum_expr_eval = expr

    def forward_quanta_weights(self, x, quanta_weights):
        """
        assume x is of shape (batch, *per_dim_features)
        """
        return self.einsum_expr_train(x, *[quanta_weights[f'{dim1} {dim2}'].to(x.dtype) for (dim1, dim2) in
                                           itertools.combinations(range(-1, -self.d_in - 1, -1), 2)])

    def forward_sum_opposite(self, x: torch.Tensor):
        assert not self.sum_mode, f'this function only works for sum_mode=False, but got {self.sum_mode=}'

        if isinstance(x, monai.data.meta_tensor.MetaTensor):
            x = x.as_tensor()

        if not self.frozen_merged:
            self.merge_frozen_weights()  # make sure the frozen weights are merged
        previous_dtype = self.weight.dtype

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.d_in > 1 and not self.merged:
            # result through raw weight
            result = F.linear(x, T(self.weight), bias=self.bias.to(previous_dtype) if self.bias is not None else None)

            # QUANTA output
            x = self.dropout(x)
            x = F.pad(x, (0, self.total_features - self.in_features), 'constant', 0.)
            x_shape = x.shape
            x = x.view(-1, *self.per_dim_features)

            # then deal with weight
            x = self.forward_quanta_weights(x, self.quanta_weights).reshape(*x_shape[:-1], self.total_features2)
            result += F.pad(x, (0, self.out_features - self.total_features2), 'constant', 0.)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias.to(previous_dtype) if self.bias is not None else None)

    def forward(self, x: torch.Tensor):
        return self.forward_sum_opposite(x)


class Conv3d(nn.Conv3d, TenVOOLayer):
    # TenVoo implemented in a Conv3d layer
    def __init__(self, in_channels: int, out_channels: int, bias: bool,
                 kernel_size: int | list | tuple = 1,
                 stride: int | list | tuple = 1,
                 padding: int | list | tuple = 1,
                 dilation: int | tuple = 1,
                 groups: int = 1,
                 tenvoo_dropout: float = 0.,
                 per_dim_features: list = None,
                 per_dim_features2: list = None,
                 merge_weights: bool = False,
                 sum_mode: bool = False,
                 initialize_mode: str = 'sum_opposite_freeze_one',  # last_layer_zero, sum_opposite_freeze_one
                 model_mode: str = 'l',  # 'l', 'q'
                 rank: int = 4,
                 **kwargs):

        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, bias=bias, **kwargs)

        # settings
        self.model_mode = model_mode
        self.initialize_mode = initialize_mode
        self.per_dim_features = per_dim_features
        self.per_dim_features2 = self.per_dim_features if per_dim_features2 is None else per_dim_features2
        self.total_features = prod(self.per_dim_features)
        self.total_features2 = prod(self.per_dim_features2)
        if self.total_features != in_channels:
            warnings.warn(
                f'per_dim_features={self.per_dim_features} does not match in_features={in_channels}, this should work but may result in downgraded performance or additional cost. Please make sure this is intended.')
        if self.total_features2 != out_channels:
            warnings.warn(
                f'per_dim_features2={self.per_dim_features2} does not match out_features={out_channels}, this should work but may result in downgraded performance or additional cost. Please make sure this is intend')
        TenVooLayer.__init__(
            self, d_in=5, d_out=5, dropout=tenvoo_dropout, merge_weights=merge_weights, sum_mode=sum_mode
        )
        # tenvoo weights
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            kD, kH, kW = kernel_size
        else:
            kD = kH = kW = kernel_size
        tenvoo_weights = self.generate_weights(kD, kH, kW, rank)
        self.tenvoo_weights = nn.ParameterDict(tenvoo_weights)
        # initialization
        if initialize_mode == 'sum_opposite_freeze_one' or initialize_mode == 'last_layer_zero':
            tenvoo_weights2 = {}
            for k, v in tenvoo_weights.items():
                tenvoo_weights2[k] = nn.Parameter(v.clone().detach(), requires_grad=False)
            self.tenvoo_weights2 = BufferDict(tenvoo_weights2)
        else:
            raise NotImplementedError
        # freezing the pre-trained weight matrix, then reset params
        self.weight.requires_grad = False
        self.reset_parameters()
        # create einsum op for training and evaluation
        self.gen_einsum_expr()

    def generate_weights(self, kD, kH, kW, rank):
        tenvoo_weights = {}
        if self.model_mode in ('l', 'tenvoo-l', 'tenvoo_l', 'tenvool'):
            tenvoo_weights['0'] = nn.Parameter(self.weight.new_zeros(self.per_dim_features[1], self.per_dim_features[0], rank, rank))
            tenvoo_weights['1'] = nn.Parameter(self.weight.new_zeros(self.per_dim_features[2], rank, rank, rank))
            tenvoo_weights['2'] = nn.Parameter(self.weight.new_zeros(kD, rank, rank, rank))
            tenvoo_weights['3'] = nn.Parameter(self.weight.new_zeros(kH, rank, rank, rank))
            tenvoo_weights['4'] = nn.Parameter(self.weight.new_zeros(kW, rank, rank, rank))
            tenvoo_weights['5'] = nn.Parameter(self.weight.new_zeros(self.per_dim_features2[2], rank, rank, rank))
            tenvoo_weights['6'] = nn.Parameter(self.weight.new_zeros(self.per_dim_features2[1], rank, rank, self.per_dim_features2[0]))
            tenvoo_weights['7'] = nn.Parameter(self.weight.new_zeros(rank, rank, rank, rank))
            tenvoo_weights['8'] = nn.Parameter(self.weight.new_zeros(rank, rank, rank, rank))
            tenvoo_weights['9'] = nn.Parameter(self.weight.new_zeros(rank, rank, rank))
        elif self.model_mode in ('q', 'tenvoo-q', 'tenvoo_q', 'tenvooq'):
            tenvoo_weights['0'] = nn.Parameter(self.weight.new_zeros(self.per_dim_features[1], self.per_dim_features[0], rank, rank))
            tenvoo_weights['1'] = nn.Parameter(self.weight.new_zeros(self.per_dim_features[2], rank, rank, rank))
            tenvoo_weights['2'] = nn.Parameter(self.weight.new_zeros(kD, rank, rank, rank))
            tenvoo_weights['3'] = nn.Parameter(self.weight.new_zeros(kH, 1, rank, rank))
            tenvoo_weights['4'] = nn.Parameter(self.weight.new_zeros(kW, rank, rank, rank))
            tenvoo_weights['5'] = nn.Parameter(self.weight.new_zeros(self.per_dim_features2[2], rank, rank, rank))
            tenvoo_weights['6'] = nn.Parameter(self.weight.new_zeros(self.per_dim_features2[1], rank, rank, self.per_dim_features2[0]))
            tenvoo_weights['7'] = nn.Parameter(self.weight.new_zeros(rank, rank, rank, rank))
            tenvoo_weights['8'] = nn.Parameter(self.weight.new_zeros(rank, rank, rank, rank))
            tenvoo_weights['9'] = nn.Parameter(self.weight.new_zeros(rank, rank, rank, rank))
        else:
            raise NotImplementedError
        return tenvoo_weights

    def reset_parameters(self):
        # reset conv3d
        nn.Conv3d.reset_parameters(self)
        if hasattr(self, 'tenvoo_weights'):
            # reset tenvoo weights
            for k, v in self.tenvoo_weights.items():
                if v.ndim == 4:
                    nn.init.kaiming_uniform_(v.view(v.shape[0] * v.shape[1], v.shape[2] * v.shape[3]), a=math.sqrt(5), nonlinearity='relu')
                else:
                    if v.shape[1] > 1:
                        nn.init.kaiming_uniform_(v.view(v.shape[0] * v.shape[1] // 2, 2 * v.shape[2]), a=math.sqrt(5), nonlinearity='relu')
                    else:
                        nn.init.kaiming_uniform_(v.view(v.shape[0], v.shape[2]), a=math.sqrt(5), nonlinearity='relu')
            # set last layer as zero
            if self.initialize_mode == 'last_layer_zero':
                nn.init.zeros_(self.tenvoo_weights[list(self.tenvoo_weights.keys())[-1]])
            # set tenvoo weigths 2 as the frozen backup of tenvoo weights
            if self.initialize_mode == 'sum_opposite_freeze_one' or self.initialize_mode == 'last_layer_zero': # define a frozon backup
                tenvoo_weights2 = {}
                for k, v in self.tenvoo_weights.items():
                    tenvoo_weights2[k] = nn.Parameter(v.clone().detach(), requires_grad=False)
                self.tenvoo_weights2 = BufferDict(tenvoo_weights2)
            else:
                raise NotImplementedError

    def gen_einsum_expr(self):
        """
        Generate the einsum expression for the tensorized weights during evaluation.
        (abc xyz ABC) denotes (i1 i2 i3 kD kH kW o3 o2 o1)
        """
        if self.model_mode in ('l', 'tenvoo-l', 'tenvoo_l', 'tenvool'):
            eq = 'bade, cefg, xghi, yijk, zklm, Cmno, BopA, fdqh, lrpn, jqr -> abcxyzABC'
        elif self.model_mode in ('q', 'tenvoo-q', 'tenvoo_q', 'tenvooq'):
            eq = 'bade, cefg, xghi, yijk, zjlm, Cmop, BpqA, oqrs, dftr, hlst -> abcxyzABC'
        else:
            raise NotImplementedError
        shapes = [self.tenvoo_weights[str(ti)].shape for ti in range(len(self.tenvoo_weights))]
        expr = oe.contract_expression(eq, *shapes, optimize='branch-all')
        self.einsum_eq = eq
        self.einsum_expr = expr

    def cal_tenvoo_weights(self):
        return self.einsum_expr(*[self.tenvoo_weights[str(ti)].to(self.weight.dtype) for ti in range(len(self.tenvoo_weights))])

    def train(self, mode: bool = True):
        nn.Conv3d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                Cout, Cin, kD, kH, kW = self.weight.shape
                full_tenvoo_weights = self.cal_tenvoo_weights().reshape(Cin, kD, kH, kW, Cout)
                full_tenvoo_weights = full_tenvoo_weights.permute(4, 0, 1, 2, 3)
                self.weight.data -= full_tenvoo_weights
                self.merged.set(False)
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                Cout, Cin, kD, kH, kW = self.weight.shape
                full_tenvoo_weights = self.cal_tenvoo_weights().reshape(Cin, kD, kH, kW, Cout)
                full_tenvoo_weights = full_tenvoo_weights.permute(4, 0, 1, 2, 3)
                self.weight.data += full_tenvoo_weights
                if not self.frozen_merged:
                    self.merge_frozen_weights()
                self.merged.set(True)

    def merge_frozen_weights(self):
        if self.frozen_merged:
            warnings.warn('The frozen weights are already merged, ignoring the request to merge the frozen weights')
        else:
            full_tenvoo_weights = -self.einsum_expr(
                *[self.tenvoo_weights2[str(ti)].to(self.weight.dtype) for ti in range(len(self.tenvoo_weights))])
            Cout, Cin, kD, kH, kW = self.weight.shape
            full_tenvoo_weights = full_tenvoo_weights.reshape(Cin, kD, kH, kW, Cout)
            full_tenvoo_weights = full_tenvoo_weights.permute(4, 0, 1, 2, 3)
            self.weight.data += full_tenvoo_weights
            self.frozen_merged.set(True)

    def unmerge_frozen_weights(self):
        if not self.frozen_merged:
            warnings.warn('The frozen weights are already unmerged, ignoring the request to unmerge the frozen weights')
        else:
            full_tenvoo_weights = -self.einsum_expr_eval(
                *[self.tenvoo_weights2[str(ti)].to(self.weight.dtype) for ti in range(len(self.tenvoo_weights2))])
            Cout, Cin, kD, kH, kW = self.weight.shape
            full_tenvoo_weights = full_tenvoo_weights.reshape(Cin, kD, kH, kW, Cout)
            full_tenvoo_weights = full_tenvoo_weights.permute(4, 0, 1, 2, 3)
            self.weight.data -= full_tenvoo_weights
            self.frozen_merged.set(False)

    def forward(self, x: torch.Tensor):
        assert not self.sum_mode, f'this function only works for sum_mode=False, but got {self.sum_mode=}'

        if isinstance(x, monai.data.meta_tensor.MetaTensor):
            x = x.as_tensor()

        if not self.frozen_merged:
            self.merge_frozen_weights()

        if not self.merged:
            Cout, Cin, kD, kH, kW = self.weight.shape
            full_tenvoo_weights = self.cal_tenvoo_weights().reshape(Cin, kD, kH, kW, Cout)
            full_tenvoo_weights = full_tenvoo_weights.permute(4, 0, 1, 2, 3)
            result = F.conv3d(x, self.weight.data + full_tenvoo_weights,
                              bias=self.bias.to(self.weight.dtype) if self.bias is not None else None,
                              stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        else:
            result = F.conv3d(x, self.weight.data,
                              bias=self.bias.to(self.weight.dtype) if self.bias is not None else None,
                              stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return result
