import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
import enum

import torch
import torch.nn as nn

from .peft_config import PeftConfig


PEFT_LAYER_KEYS = ["locon_", "lora_", "loha_", "lokr_", "quanta_", "tenvoo_"]

class PeftType(str, enum.Enum):
    LORA = "LOCON"
    LOCON = "LOCON"
    LOHA = "LOHA"
    LOKR = "LOKR"
    OFT = "OFT"
    QUANTA = "QUANTA"
    TENVOO = "TENVOO"

@dataclass
class PeftConvConfig(PeftConfig):
    target_modules: Optional[Union[List[str], str]] = field(default=None, metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['.*to_q', '.*to_v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'"}
        )
    rank: int = field(default=4, metadata={"help": "rank of low rank layers"})
    bias: str = field(default='lora_only', metadata={"help": "mode of gradients for bias"})  # none all lora_only
    requires_full_weights_grad: bool = field(default=False, metadata={"help": "whether to fine-tune the non-low-rank layers or not"})
    exclude_first_last_conv: bool = field(default=True, metadata={"help": "whether to exclude first and final convolutional layer or not"})


class PeftLayer:
    def __init__(self, r: int, dropout: float, merge_weights: bool):
        self.r = r
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class PeftModel(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_lora_layernorm_cls_trainable(self.model, self.peft_config.bias, self.peft_config.requires_full_weights_grad)
        self.forward = self.model.forward 

        print("The settings of your model are:\n", config)

    def _find_and_replace(self):
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            # a. search for modules. iterate over model, to find target modules
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.search(self.peft_config.target_modules, key) is not None
            else:
                target_module_found = any(re.search(target_key, key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                
                # b. build new modules
                parent, target, target_name = self._get_submodules(key)
                if isinstance(target, torch.nn.Linear):
                    bias = target.bias is not None
                    kwargs = self.get_linear_args(target)
                    new_module = self.get_new_linear_module(target.in_features, target.out_features, bias=bias, **kwargs)
                    self._replace_module(parent, target_name, new_module, target)
                elif isinstance(target, torch.nn.Conv3d):
                    # skip input & output convolutional layers
                    if (target.in_channels == 1 or target.out_channels == 1) and self.peft_config.exclude_first_last_conv:
                        continue
                    bias = target.bias is not None
                    kwargs = self.get_conv_args(target)
                    new_module = self.get_new_conv_module(target.in_channels, target.out_channels, target.kernel_size, \
                            target.stride, target.padding, bias, target.dilation, **kwargs)
                    self._replace_module(parent, target_name, new_module, target)
                else:
                    raise NotImplementedError

        if not is_target_modules_in_base_model:
            raise ValueError(f"Target modules {self.peft_config.target_modules} not found in the base model. "
                             f"Please check the target modules and try again.")
    
    def get_linear_args(self, target):
        raise NotImplementedError
    
    def get_conv_args(self, target):
        raise NotImplementedError
    
    def get_new_linear_module(self, **kwargs):
        raise NotImplementedError

    def get_new_conv_module(self, **kwargs):
        raise NotImplementedError

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        # replace
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        new_module.weight.requires_grad = False
        if old_module.bias is not None:
            new_module.bias = old_module.bias
            new_module.bias.requires_grad = False
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if any(key in name for key in PEFT_LAYER_KEYS):
                module.to(old_module.weight.device)
            if 'bias' in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, PeftLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


def mark_lora_layernorm_cls_trainable(model: nn.Module, bias: str = "lora_only", requires_full_weights_grad: bool = False) -> None:
    if not requires_full_weights_grad:
        # for all layers, set requires_grad = False for all layer escept quanta layers
        for n, p in model.named_parameters():
            if all(key not in n for key in PEFT_LAYER_KEYS):
                p.requires_grad = False

        # for bias, default lora_only
        if bias == "none":
            for m in model.modules():
                if isinstance(m, PeftLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = False
        elif bias == "all":
            for n, p in model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True 
        elif bias == "lora_only":
            for m in model.modules():
                if isinstance(m, PeftLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError  # mark layer-norm trainable

        # for norm
        for n, p in model.named_parameters():
            if "norm" in n.lower():
                p.requires_grad = True


class MergeBuffer(nn.Module):
    def __init__(self, default=False):
        super(MergeBuffer, self).__init__()
        self.register_buffer('merged', torch.tensor(default))  # to keep track if the trainable weights are merged

    def __bool__(self):
        return self.merged.item()

    def set(self, value):
        self.merged.fill_(value)
