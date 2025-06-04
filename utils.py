import random
import torch
import numpy as np
import torch.nn as nn

from peft import PeftLayer

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def peft2nnlayer(module):
    if isinstance(module, PeftLayer):
        with torch.no_grad():
            weight = module.weight.clone()
            bias = module.bias.clone() if module.bias is not None else None
            device = module.weight.device
            if isinstance(module, nn.Conv3d):
                layer = nn.Conv3d(
                    module.in_channels, module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    bias=bias is not None
                )
            elif isinstance(module, nn.Linear):
                layer = nn.Linear(
                    module.in_features, module.out_features,
                    bias=bias is not None
                )
            else:
                return module
            layer = layer.to(device)
            layer.weight.copy_(weight)
            if bias is not None:
                layer.bias.copy_(bias)
            return layer
    else:
        return module

def peft2nnmodel(model):
    for name, module in model.named_children():
        new_module = peft2nnlayer(module)
        if new_module is not module:
            setattr(model, name, new_module)
        else:
            peft2nnmodel(module)
    return model

def save_peft(model, path, model_type='tenvoo-l', info=False):
    # define target keys for saving
    model_type = model_type.lower()
    assert model_type in ('lora', 'locon', 'lokr', 'loha', 'tenvoo-l', 'tenvoo-q'), f'Unsupported model_type = {model_type}'
    keys = {
        'lora': ['lora'],
        'locon': ['lora'],
        'lokr': ['lokr'],
        'loha': ['loha'],
        'tenvoo-l': ['tenvoo', 'quanta'],
        'tenvoo-q': ['tenvoo', 'quanta']
    }[model_type]
    # search and save
    state_dict = {}
    for module_name, module in model.named_modules():
        if isinstance(module, PeftLayer):
            # Save parameters
            for param_name, param in module.named_parameters(recurse=True):
                full_name = f"{module_name}.{param_name}"
                if any(k in full_name for k in keys):
                    if info:
                        print(f"[SAVE]: {full_name} shaped {param.shape}")
                    state_dict[full_name] = param.detach().cpu()
            # Save buffers (e.g., tenvoo_weights2)
            for buffer_name, buffer in module.named_buffers(recurse=True):
                full_name = f"{module_name}.{buffer_name}"
                if any(k in full_name for k in keys):
                    if info:
                        print(f"[SAVE]: {full_name} shaped {buffer.shape}")
                    state_dict[full_name] = buffer.detach().cpu()
    torch.save(state_dict, path)

def load_peft(model, path):
    # load
    state_dict = torch.load(path, map_location='cpu')
    print("[INFO] Loaded", len(state_dict), f"parameters from PEFT checkpoint {path}")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    """
    if missing_keys:
        print("[WARN] Missing keys:", missing_keys[:10], "..." if len(missing_keys) > 10 else "")
    if unexpected_keys:
        print("[WARN] Unexpected keys:", unexpected_keys[:10], "..." if len(unexpected_keys) > 10 else "")
    # """
    # reset merged and frozen_merged
    for module in model.modules():
        if isinstance(module, PeftLayer):
            module.merged.set(False)
            if hasattr(module, 'frozen_merged'):
                module.frozen_merged.set(False)
    return model
    
