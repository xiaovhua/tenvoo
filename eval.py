# https://github.com/batmanlab/HA-GAN/blob/master/evaluation/fid_score.py
# !/usr/bin/env python3

import os
import ast
import glob
import argparse
import torch
import nibabel as nib
import numpy as np
from scipy import linalg
from pytorch_msssim import ms_ssim
from typing import OrderedDict
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from generative.inferers import DiffusionInferer

from dataset import DDPMDataset, default_transform, GLOB_PATHS
from peft import (LoConConfig, LoConModel, LokrConfig, LokrModel, LohaConfig,
                  LohaModel, TenVOOConfig, TenVOOModel, TENVOO_LIST)
from ddpm_unet import DDPM_LAYERS, init_diffusion_unet
from utils import seed_everything, load_peft, peft2nnmodel
from med3d import resnet50


############################################ Med3D-Resnet50 ############################################

def trim_state_dict_name(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

def load_resnet50(path):
    model = resnet50(shortcut_type='B')
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten())  # (N, 512)
    # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    ckpt = torch.load(path)
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt)
    # model = nn.DataParallel(model)
    print(f"Feature extractor weights are loaded from {path}")
    return model

############################################ Features ############################################

def cal_feats_from_dataloader(model, data_loader, args):
    model.eval()
    pred_arr = np.empty((len(data_loader.dataset), 2048), dtype=np.float32)
    with torch.no_grad():
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=args.use_amp):
            for i, batch in enumerate(data_loader):
                if i % 10 == 0:
                    print(f'\rPropagating batch {i}', end='', flush=True)
                inputs = batch.float().to(next(model.parameters()).device)
                pred = model(inputs)
                pred_arr[i * args.batch_size:(i + 1) * args.batch_size] = pred.cpu().numpy()
    print('\ndone')
    return pred_arr


def generate_and_encode(unet, inferer, encoder, args):
    output_dir = os.path.join(args.output_dir, f"images_{args.dataset_name}_{args.ft_mode}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # generate and save images
    with torch.no_grad():
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=args.use_amp):
            for i in range(args.num_samples):
                save_path = os.path.join(output_dir, f'{i}.nii.gz')
                noise = torch.randn(args.img_size).cuda()
                synt = inferer.sample(
                    input_noise=noise, diffusion_model=unet, scheduler=inferer.scheduler
                )
                synt = synt.detach().cpu().numpy()  # (1, 1, H, W, D)
                while synt.ndim > 3:
                    synt = synt[0]
                nib.save(nib.Nifti1Image(synt, np.eye(4)), save_path)
    # load and encode images
    pred_arr = np.empty((args.num_samples, 2048))
    with torch.no_grad():
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=args.use_amp):
            for i in range(args.num_samples):
                # load
                save_path = os.path.join(output_dir, f'{i}.nii.gz')
                synt = nib.load(save_path).get_fdata(dtype=np.float32)
                # double check the dimension
                if synt.ndim == 5:
                    nib.save(nib.Nifti1Image(synt[0][0], np.eye(4)), save_path)
                    synt = nib.load(save_path).get_fdata(dtype=np.float32)
                if synt.ndim == 4:
                    nib.save(nib.Nifti1Image(synt[0], np.eye(4)), save_path)
                    synt = nib.load(save_path).get_fdata(dtype=np.float32)
                synt = np.expand_dims(synt, axis=(0, 1))
                # encode
                pred = encoder(torch.tensor(synt).to(encoder.device))  # (1, 2048)
                pred_arr[i] = pred[0].cpu().numpy()
    return pred_arr


############################################ FID3D ############################################

def post_process(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular, add offset to avoid overflow
    covmean, _ = linalg.sqrtm((sigma1).dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(unet, inferer, encoder, train_loader, valid_loader, args, save=True, mode='real'):
    output_dir = os.path.join(args.output_dir, f"metrics")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # 1. encode images from valid_loader
    if not os.path.exists(os.path.join(output_dir, f"{args.dataset_name}_val.npy")):
        feats_val = cal_feats_from_dataloader(encoder, valid_loader, args)
        if save:
            np.save(os.path.join(output_dir, f"{args.dataset_name}_val.npy"), feats_val)
    else:
        feats_val = np.load(os.path.join(output_dir, f"{args.dataset_name}_val.npy"))
    # 2. encode targeted images
    if mode == 'real':
        # encode images from train_loader
        if not os.path.exists(os.path.join(output_dir, f"{args.dataset_name}_train.npy")):
            feats_train = cal_feats_from_dataloader(encoder, train_loader, args)
            if save:
                np.save(os.path.join(output_dir, f"{args.dataset_name}_train.npy"), feats_train)
        else:
            feats_train = np.load(os.path.join(output_dir, f"{args.dataset_name}_train.npy"))
        feats_target = feats_train
    elif mode == 'gen':
        # generate images and then encode them
        if not os.path.exists(os.path.join(output_dir, f"{args.dataset_name}_{args.ft_mode}_gen.npy")):
            feats_gen = generate_and_encode(unet, inferer, encoder, args)
            if save:
                np.save(os.path.join(output_dir, f"{args.dataset_name}_{args.ft_mode}_gen.npy"), feats_gen)
        else:
            feats_gen = np.load(os.path.join(output_dir, f"{args.dataset_name}_{args.ft_mode}_gen.npy"))
        feats_target = feats_gen
    else:
        raise NotImplementedError
    # 3. calculate fid score
    m1, s1 = post_process(feats_val)
    m, s = post_process(feats_target)
    fid_value = calculate_frechet_distance(m1, s1, m, s)
    print(f"\nDataset: {args.dataset_name}, Method: {args.ft_mode}, FID score: {fid_value}")
    return fid_value


############################################ MMD ############################################

def calculate_mmd(unet, inferer, encoder, train_loader, valid_loader, args, save=True, mode='real'):
    # install from https://github.com/josipd/torch-two-sample/
    from torch_two_sample.statistics_diff import MMDStatistic

    output_dir = os.path.join(args.output_dir, f"metrics")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # 1. encode images from valid_loader
    if not os.path.exists(os.path.join(output_dir, f"{args.dataset_name}_val.npy")):
        feats_val = cal_feats_from_dataloader(encoder, valid_loader, args)
        if save:
            np.save(os.path.join(output_dir, f"{args.dataset_name}_val.npy"), feats_val)
    else:
        feats_val = np.load(os.path.join(output_dir, f"{args.dataset_name}_val.npy"))
    # 2. encode targeted images
    if mode == 'real':
        # encode images from train_loader
        if not os.path.exists(os.path.join(output_dir, f"{args.dataset_name}_train.npy")):
            feats_train = cal_feats_from_dataloader(encoder, train_loader, args)
            if save:
                np.save(os.path.join(output_dir, f"{args.dataset_name}_train.npy"), feats_train)
        else:
            feats_train = np.load(os.path.join(output_dir, f"{args.dataset_name}_train.npy"))
        feats_target = feats_train
    elif mode == 'gen':
        # generate images and then encode them
        if not os.path.exists(os.path.join(output_dir, f"{args.dataset_name}_{args.ft_mode}_gen.npy")):
            feats_gen = generate_and_encode(unet, inferer, encoder, args)
            if save:
                np.save(os.path.join(output_dir, f"{args.dataset_name}_{args.ft_mode}_gen.npy"), feats_gen)
        else:
            feats_gen = np.load(os.path.join(output_dir, f"{args.dataset_name}_{args.ft_mode}_gen.npy"))
        feats_target = feats_gen
    else:
        raise NotImplementedError
    # 3. calculate mmd score
    mmd = MMDStatistic(feats_val.shape[0], feats_target.shape[0])
    sample_1 = torch.from_numpy(feats_val)
    sample_2 = torch.from_numpy(feats_target)
    distances = torch.cdist(sample_1, sample_2, p=2)
    alpha_median = torch.median(distances)
    test_statistics, ret_matrix = mmd(sample_1, sample_2, alphas=[alpha_median], ret_matrix=True)
    print(f"\nDataset: {args.dataset_name}, Method: {args.ft_mode}, FID score: {test_statistics.item()}")
    return test_statistics.item()


############################################ MS-SSIM ############################################

def calculate_msssim(unet, inferer, encoder, train_loader, valid_loader, args, mode='real'):
    # re-generate images (if not exist)
    if mode == 'gen':
        output_dir = os.path.join(args.output_dir, f"images_{args.dataset_name}_{args.ft_mode}")
        re_generate = False
        for i in range(args.num_samples):
            if not os.path.exists(os.path.join(output_dir, f'{i}.nii.gz')):
                re_generate = True
                break
        if re_generate:
            _ = generate_and_encode(unet, inferer, encoder, args)
        paths = [p for p in os.listdir(output_dir) if p.endswith('.nii.gz')]

    msssim = 0
    cnt = 0
    with torch.no_grad():
        if mode == 'real':
            for imgs_val, imgs_train, __ in zip(valid_loader, train_loader, range(args.num_samples)):
                cnt += 1
                with torch.autocast(enabled=True, device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    imgs_val = imgs_val.to(device)
                    imgs_train = imgs_train.to(device)
                    while imgs_val.ndim < 5:
                        imgs_val = imgs_val.unsqueeze(0)
                    while imgs_train.ndim < 5:
                        imgs_train = imgs_train.real.unsqueeze(0)
                    msssim += ms_ssim(imgs_val, imgs_train, data_range=1., win_size=5 if args.dataset_name == 'brats' else 7, size_average=False)

        elif mode == 'gen':
            for imgs_val, p, __ in zip(valid_loader, paths, range(args.num_samples)):
                cnt += 1
                with torch.autocast(enabled=True, device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    synt = torch.tensor(nib.load(p).get_fdata(dtype=np.float32)).to(device)
                    imgs_val = imgs_val.to(device)
                    while synt.ndim < 5:
                        synt = synt.unsqueeze(0)
                    while imgs_val.ndim < 5:
                        imgs_val = imgs_val.unsqueeze(0)
                    msssim += ms_ssim(imgs_val, synt, data_range=1., win_size=5 if args.dataset_name == 'brats' else 7, size_average=False)

        else:
            raise NotImplementedError

    msssim = msssim.mean().item() / cnt
    print(f"\nDataset: {args.dataset_name}, Method: {args.ft_mode}, SSIM score: {msssim}")
    return msssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset_root', required=True, type=str, help='path to your MRI images')
    parser.add_argument('--mri_resolution', default='[(160, 224, 160)]', type=str,
                        help='spatial resolution of mri scans')
    parser.add_argument('--scale_ratio', default=1., type=float, help='downsample ratio for mri scans')
    parser.add_argument('--dataset_name', type=str, default='ukb', choices={'ukb', 'adni', 'brats', 'ppmi'},
                        help='name of the dataset')
    # saving
    parser.add_argument('--output_dir', required=True, type=str, help='path to save the results')
    parser.add_argument('--unet_ckpt', required=True, type=str, help='path to the saved unet checkpoint')
    parser.add_argument('--peft_ckpt', required=True, type=str, help='path to the saved peft checkpoint')
    parser.add_argument('--med3d_ckpt', required=True, type=str, default='./resnet_50.pth', help='path to the saved medicalnet checkpoint')

    # model
    parser.add_argument('--ddpm_steps', default=1000, type=int, help='total timesteps for ddpm')
    parser.add_argument('--scheduler_type', choices={'ddpm', 'ddim'}, default='ddpm', type=str, help='approach to sample')
    # peft
    parser.add_argument('--ft_mode', default='ft', type=str,
                        choices={'ff', 'lora', 'locon', 'lokr', 'loha', 'tenvoo-l', 'tenvoo-q'}, help='type of PEFT way')
    parser.add_argument('-r', '--rank', default=8, type=int, help='rank for quanta or oft')
    parser.add_argument('--target_modules', type=str, default=None, help='name of modules to use PEFT')
    # others
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for your dataloader')
    parser.add_argument('--batch_size', default=128, type=int, help='number of your batch size')
    parser.add_argument('--num_samples', type=int, default=100, help='number of generated samples for evaluation')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--use_amp', choices={0, 1}, default=1, type=int,
                        help='whether to use automatic mixed precision')

    args = parser.parse_args()
    args.use_amp = args.use_amp == 1
    args.mri_resolution = ast.literal_eval(args.mri_resolution)[0]
    args.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    ### To use the same testing samples, please use the same random seed as during training ###
    seed_everything(args.seed)

    # dataset
    if args.dataset_name == 'brats':
        args.dataset_root = os.path.join(args.dataset_root, *GLOB_PATHS[args.dataset_name])
        if os.path.exists(os.path.join(args.dataset_root, 'clean_brats.txt')): # filter brats mri scans
            with open(os.path.join(args.dataset_root, 'clean_brats.txt'), 'r', encoding='utf-8') as fr:
                clean_files = [f.strip() for f in fr.readlines()]
            mir_list = [r for r in glob.glob(args.dataset_root) if r.replace('\\', '/').split('/')[-1] in clean_files]
        else:
            mir_list = [r for r in glob.glob(args.dataset_root)]
        dataset = DDPMDataset(root=mir_list, transform=default_transform(spatial_size=args.mri_resolution, scale_ratio=args.scale_ratio))
    else:
        args.dataset_root = os.path.join(args.dataset_root, *GLOB_PATHS[args.dataset_name])
        dataset = DDPMDataset(root=args.dataset_root, transform=default_transform(spatial_size=args.mri_resolution, scale_ratio=args.scale_ratio))
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    trainset, validset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(dataset=trainset, num_workers=args.num_workers, batch_size=args.batch_size, persistent_workers=True, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=validset, num_workers=args.num_workers, batch_size=args.batch_size, persistent_workers=True, pin_memory=True)
    args.img_size = (1, ) + trainset[0].shape

    # model
    unet = init_diffusion_unet(1, args.unet_ckpt).to(device)
    if args.scheduler_type == 'ddpm':
        scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_steps,
            schedule='scaled_linear_beta',
            beta_start=0.0005,
            beta_end=0.0195,
        )
        scheduler.set_timesteps(num_inference_steps=args.ddpm_steps)
    elif args.scheduler_type == 'ddim':
        scheduler = DDIMScheduler(
            num_train_timesteps=args.ddpm_steps,
            schedule='scaled_linear_beta',
            beta_start=0.0005,
            beta_end=0.0195,
            clip_sample=False
        )
        scheduler.set_timesteps(num_inference_steps=args.ddpm_steps // 4)
    else:
        raise NotImplementedError
    inferer = DiffusionInferer(scheduler)
    scaler = GradScaler(enabled=args.use_amp)
    total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    # peft
    # TODO: possible settings for PEFT
    # # https://github.com/pytorch/pytorch/pull/60451
    # # disable tf32 to avoid calculation error
    # torch.backends.cudnn.allow_tf32 = False
    # torch.set_float32_matmul_precision('high')
    if args.ft_mode == 'ff': # full fine-tuning:
        unet.eval()
    else:
        assert args.target_modules is not None, f"Using ft_mode={args.ft_mode}, but got target_modules=None"
        target_modules = [DDPM_LAYERS[t] for t in args.target_modules.split(',')]
        if args.ft_mode == 'locon' or args.ft_mode == 'lora':
            config = LoConConfig(
                merge_weights=True, target_modules=target_modules, rank=args.rank, alpha=0.0,
            )
            unet = LoConModel(config, unet).to(device)
        if args.ft_mode == 'lokr':
            config = LokrConfig(
                merge_weights=True, target_modules=target_modules, rank=args.rank, alpha=0.0, factor=-1
            )
            unet = LokrModel(config, unet).to(device)
        if args.ft_mode == 'loha':
            config = LohaConfig(
                merge_weights=True, target_modules=target_modules, rank=args.rank, alpha=0.0
            )
            unet = LohaModel(config, unet).to(device)
        if args.ft_mode == 'tenvoo-l' or args.ft_mode == 'tenvoo-q':
            D = 3  # for Linear, only 3 are supported now
            model_mode = args.ft_mode.replace('_', '-').split('-')[-1]
            config = TenVOOConfig(
                d_in=D, d_out=D, per_dim_list=TENVOO_LIST, merge_weights=True, target_modules=target_modules, sum_mode=False, dropout=0.0, rank=args.rank, model_mode=model_mode
            )
            unet = TenVOOModel(config, unet).to(device)
        for n, p in unet.named_parameters():
            if p.requires_grad:
                print(n)
        print("Low-rank layers and their names:")
        unet = load_peft(unet, args.peft_ckpt)
        unet.eval()
        # transform peft model to torch.nn model
        unet = peft2nnmodel(unet)
    train_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(
        f"The raw UNet has {(total_params / 1000 / 1000):.4}M parameters, while only {(train_params / 1000 / 1000):.4}M ({(train_params / total_params):.4}%) are used for fine-tuning."
    )

    # encoder from https://github.com/Tencent/MedicalNet
    encoder = load_resnet50(args.med3d_ckpt)
    encoder.eval()
    encoder.to(device)

    # fid
    fid_real = calculate_fid(unet, inferer, encoder, train_loader, valid_loader, args, save=True, mode='real')
    fid_gen = calculate_fid(unet, inferer, encoder, train_loader, valid_loader, args, save=True, mode='gen')

    # mmd
    mmd_real = calculate_mmd(unet, inferer, encoder, train_loader, valid_loader, args, save=True, mode='real')
    mmd_gen = calculate_mmd(unet, inferer, encoder, train_loader, valid_loader, args, save=True, mode='gen')

    # ssim
    # set batch_size = 1
    del train_loader, valid_loader
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        num_workers=args.num_workers,
        batch_size=1,
        drop_last=False,
        persistent_workers=True,
        shuffle=False,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=validset,
        num_workers=args.num_workers,
        batch_size=1,
        drop_last=False,
        persistent_workers=True,
        shuffle=False,
        pin_memory=True
    )

    # msssim
    ms_real = calculate_msssim(unet, inferer, encoder, train_loader, valid_loader, args, mode='real')
    ms_gen = calculate_msssim(unet, inferer, encoder, train_loader, valid_loader, args, mode='gen')

    print(args)
    print(
        f'FID_real: {fid_real}, FID_gen: {fid_gen}, MMD_real: {mmd_real}, MMD_gen: {mmd_gen}, MSSSIM_real: {ms_real}, MSSSIM_gen: {ms_gen}'
    )

