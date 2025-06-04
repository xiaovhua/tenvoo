import glob
import os
import ast
import argparse
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer

from dataset import DDPMDataset, default_transform, GLOB_PATHS
from peft import (LoConConfig, LoConModel, LokrConfig, LokrModel, LohaConfig,
                  LohaModel, TenVOOConfig, TenVOOModel, TENVOO_LIST)
from utils import seed_everything, save_peft
from ddpm_unet import DDPM_LAYERS, init_diffusion_unet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset_root', required=True, type=str, help='path to your MRI images')
    parser.add_argument('--mri_resolution', default='[(160, 224, 160)]', type=str, help='spatial resolution of mri scans')
    parser.add_argument('--scale_ratio', default=1., type=float, help='downsample ratio for mri scans')
    parser.add_argument('--dataset_name', type=str, default='ukb', choices={'ukb', 'adni', 'brats', 'ppmi'}, help='name of the dataset')
    # saving
    parser.add_argument('--output_dir', required=True, type=str, help='path to save your models')
    parser.add_argument('--unet_ckpt', required=True, type=str, help='path to the saved unet checkpoint')
    # model
    parser.add_argument('--ddpm_steps', default=1000, type=int, help='total timesteps for ddpm')
    # peft
    parser.add_argument('--ft_mode', default='ft', type=str,
                        choices={'ff', 'lora', 'locon', 'lokr', 'loha', 'tenvoo-l', 'tenvoo-q'}, help='type of PEFT way')
    parser.add_argument('-r', '--rank', default=8, type=int, help='rank for quanta or oft')
    parser.add_argument('--target_modules', type=str, default=None, help='name of modules to use PEFT')
    parser.add_argument('--joint', type=int, default=0, choices={0, 1}, help='whether to use joint training when using PEFT')
    parser.add_argument('--peft_bias', type=str, default="lora_only", choices={"lora_only", "none", "all"}, help='whether to train the bias vectors in the model')
    # only for tenvoo
    parser.add_argument('--initialize_mode', type=str, default='sum_opposite_freeze_one',
                        choices={'sum_opposite_freeze_one', 'last_layer_zero'}, help='way to initialize core tensors')
    # training
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for your dataloader')
    parser.add_argument('--accu_steps', default=4, type=int, help='gradient accumulate steps')
    parser.add_argument('--n_epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--warmup', default=1, type=int, help='number of warmup epochs for autoencoder')
    parser.add_argument('--batch_size', default=1, type=int, help='number of your batch size')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--loss_mode', default='mse', type=str, choices={'mse', 'mae'}, help='loss function')
    # others
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--use_amp', choices={0, 1}, default=1, type=int, help='whether to use automatic mixed precision')

    # settings
    args = parser.parse_args()
    args.use_amp = args.use_amp == 1
    args.joint = args.joint == 1
    args.mri_resolution = ast.literal_eval(args.mri_resolution)[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.output_dir = os.path.join(args.output_dir,  f'{args.dataset_name}-{args.ft_mode}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    seed_everything(args.seed)

    # dataset
    if args.dataset_name == 'brats':
        if os.path.exists(os.path.join(args.dataset_root, 'clean_brats.txt')): # filter brats mri scans
            with open(os.path.join(args.dataset_root, 'clean_brats.txt'), 'r', encoding='utf-8') as fr:
                clean_files = [f.strip() for f in fr.readlines()]
        args.dataset_root = os.path.join(args.dataset_root, *GLOB_PATHS[args.dataset_name])
        mir_list = [r for r in glob.glob(args.dataset_root) if r.replace('\\', '/').split('/')[-1] in clean_files]
        dataset = DDPMDataset(root=mir_list, transform=default_transform(spatial_size=args.mri_resolution, scale_ratio=args.scale_ratio))
    else:
        args.dataset_root = os.path.join(args.dataset_root, *GLOB_PATHS[args.dataset_name])
        dataset = DDPMDataset(root=args.dataset_root, transform=default_transform(spatial_size=args.mri_resolution, scale_ratio=args.scale_ratio))
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    trainset, validset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(dataset=trainset, num_workers=args.num_workers, batch_size=args.batch_size, persistent_workers=True, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=validset, num_workers=args.num_workers, batch_size=args.batch_size, persistent_workers=True, pin_memory=True)

    # model
    unet = init_diffusion_unet(channels=1, checkpoints_path=args.unet_ckpt).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_steps,
        schedule='scaled_linear_beta',
        beta_start=0.0005,
        beta_end=0.0195,
    )
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
        pass
    else:
        assert args.target_modules is not None, f"Using ft_mode={args.ft_mode}, but got target_modules=None"
        target_modules = [DDPM_LAYERS[t] for t in args.target_modules.split(',')]
        if args.ft_mode == 'locon' or args.ft_mode == 'lora':
            config = LoConConfig(
                merge_weights=True, target_modules=target_modules, bias=args.peft_bias, rank=args.rank, alpha=0.0,
                requires_full_weights_grad=args.joint, exclude_first_last_conv=True
            )
            unet = LoConModel(config, unet).to(device)
        if args.ft_mode == 'lokr':
            config = LokrConfig(
                merge_weights=True, target_modules=target_modules, bias=args.peft_bias, rank=args.rank, alpha=0.0,
                factor=-1, requires_full_weights_grad=args.joint, exclude_first_last_conv=True
            )
            unet = LokrModel(config, unet).to(device)
        if args.ft_mode == 'loha':
            config = LohaConfig(
                merge_weights=True, target_modules=target_modules, bias=args.peft_bias, rank=args.rank, alpha=0.0,
                requires_full_weights_grad=args.joint, exclude_first_last_conv=True
            )
            unet = LohaModel(config, unet).to(device)
        if args.ft_mode == 'tenvoo-l' or args.ft_mode == 'tenvoo-q':
            D = 3  # for Linear, only 3 are supported now
            model_mode = args.ft_mode.split('-')[-1]
            config = TenVOOConfig(
                d_in=D, d_out=D, per_dim_list=TENVOO_LIST, merge_weights=True, target_modules=target_modules,
                model_mode=model_mode, initialize_mode=args.initialize_mode,
                sum_mode=False, bias=args.peft_bias, dropout=0.0, rank=args.rank, requires_full_weights_grad=args.joint,
                exclude_first_last_conv=True
            )
            unet = TenVOOModel(config, unet).to(device)
        # info
        for n, p in unet.named_parameters():
            if p.requires_grad:
                print(n)
        print("Low-rank layers and their names:")
    train_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(
        f"The raw UNet has {(total_params / 1000 ** 2):.4}M parameters, "
        f"while only {(train_params / 1000 ** 2):.4}M ({(train_params / total_params * 100):.4}%) are used for fine-tuning."
    )

    # train
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    print(
        f'MRI images are of shape {trainset[0].shape}. Use {len(trainset)} for training, '
        f'{len(validset)} for evaluation. Unet model is loaded from {args.unet_ckpt}.'
    )
    for epoch in range(args.n_epochs):
        epoch_loss = 0
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        train_bar.set_description(f"Epoch {epoch}")
        valid_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), ncols=70)
        valid_bar.set_description(f"Epoch {epoch}")

        # train
        for step, imgs in train_bar:
            unet.train()
            # train
            imgs = imgs.to(device)
            # TODO
            imgs = imgs.as_tensor()
            with autocast(enabled=args.use_amp, dtype=torch.float16):
                noise = torch.randn_like(imgs).to(device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (imgs.shape[0],), device=imgs.device
                ).long()
                target = noise
                noise_pred = inferer(
                    inputs=imgs, diffusion_model=unet, noise=noise, timesteps=timesteps
                )
                loss = F.mse_loss(noise_pred.float(), target.float()) if args.loss_mode == 'mse' \
                    else F.l1_loss(noise_pred.float(), target.float())
            scaler.scale(loss).backward()
            if (step + 1) % args.accu_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            train_bar.set_postfix({'loss': epoch_loss / (step + 1)})
            # # info memory usage
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # valid
        if valid_loader is not None and epoch + 1 == args.n_epochs:
            v_loss = 0
            for valid_step, valid_imgs in valid_bar:
                unet = unet.eval()
                with autocast(enabled=args.use_amp):
                    with torch.no_grad():
                        valid_imgs = valid_imgs.to(device)
                        valid_noise = torch.randn_like(valid_imgs).to(device)
                        valid_timesteps = torch.randint(
                            0,
                            inferer.scheduler.num_train_timesteps,
                            (valid_imgs.shape[0],),
                            device=valid_imgs.device
                        ).long()
                        valid_target = valid_noise
                        # forward
                        valid_noise_pred = inferer(
                            inputs=valid_imgs, diffusion_model=unet, noise=valid_noise, timesteps=valid_timesteps
                        )
                        valid_loss = F.mse_loss(valid_noise_pred.float(), valid_target.float()) if args.loss_mode == 'mse' \
                            else F.l1_loss(valid_noise_pred.float(), valid_target.float())
                v_loss += valid_loss.detach().item()
                valid_bar.set_postfix({'loss': v_loss / (valid_step + 1)})
        # save checkpoints
        save_peft(unet, os.path.join(args.output_dir, f'ep{epoch + 1}.pth'), args.ft_mode)
    # save the last checkpoint
    save_peft(unet, os.path.join(args.output_dir, f'last.pth'), args.ft_mode)

    # check memory peak
    peak_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    print(f"[CUDA] Peak Allocated: {peak_allocated:.2f} MB")
    print(f"[CUDA] Peak Reserved:  {peak_reserved:.2f} MB")

