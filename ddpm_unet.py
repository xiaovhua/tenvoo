import os
import torch
import torch.nn as nn
from typing import Optional
from generative.networks.nets import DiffusionModelUNet


DDPM_LAYERS = {
    "conv1": ".*resnets.*.conv1.conv",
    "conv2": ".*resnets.*.conv2.conv",
    "to_q": ".*attentions.*.to_q",
    "to_v": ".*attentions.*.to_v",
    "time_emb": "time_embed.2",
    "time_proj": ".*resnets.*.time_emb_proj",
    # not used in our work
    "attn_proj": ".*attentions.*.*.attn_proj",
    "to_k": ".*attentions.*.to_k",
    "skip": ".*resnets.*.skip_connection.conv",
    "ds": ".*downsampler.op.conv",
    "us": ".*upsampler.conv.conv",
}


def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        network.load_state_dict(torch.load(checkpoints_path))
        print(f"UNet weights are loaded from {checkpoints_path}")
    return network


def init_diffusion_unet(channels: int, checkpoints_path: Optional[str] = None) -> nn.Module:
    diffusion_unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=channels,
        out_channels=channels,
        num_res_blocks=2,
        num_channels=(32, 32, 64, 128, 256, 512),
        attention_levels=(False, False, False, False, True, True),
        norm_num_groups=32,
        norm_eps=1e-6,
        num_head_channels=(0, 0, 0, 0, 256, 512)
    )
    return load_if(checkpoints_path, diffusion_unet)
