import os
import argparse
import pickle
import ast
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from tqdm import tqdm

from utils import seed_everything
from dataset import DDPMDataset, default_transform, GLOB_PATHS
from ddpm_unet import init_diffusion_unet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset_root', required=True, type=str, help='path to your MRI images')
    parser.add_argument('--mri_resolution', default='[(160, 224, 160)]', type=str, help='spatial resolution of mri scans')
    parser.add_argument('--scale_ratio', default=1., type=float, help='downsample ratio for mri scans')
    parser.add_argument('--dataset_name', type=str, default='ukb', choices={'ukb',}, help='name of the dataset')
    # saving
    parser.add_argument('--output_dir', required=True, type=str, help='path to save your models')
    parser.add_argument('--unet_ckpt', type=str, default=None, help='path to the saved unet checkpoint')
    # model
    parser.add_argument('--ddpm_steps', default=1000, type=int, help='total timesteps for ddpm')
    # training
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for your dataloader')
    parser.add_argument('--accu_steps', default=4, type=int, help='gradient accumulate steps')
    parser.add_argument('--n_epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--warmup', default=1, type=int, help='number of warmup epochs for autoencoder')
    parser.add_argument('--batch_size', default=1, type=int, help='number of your batch size')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--loss_mode', default='mse', type=float, choices={'mse', 'mae'}, help='loss function')
    # others
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--use_amp', choices={0, 1}, default=1, type=int, help='whether to use automatic mixed precision')

    # settings
    args = parser.parse_args()
    args.use_amp = args.use_amp == 1
    args.dataset_root = os.path.join(args.dataset_root, *GLOB_PATHS[args.dataset_name])
    args.mri_resolution = ast.literal_eval(args.mri_resolution)[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    seed_everything(args.seed)

    # dataset
    dataset = DDPMDataset(root=args.dataset_root, transform=default_transform(spatial_size=args.mri_resolution, scale_ratio=args.scale_ratio))
    train_size = int(len(dataset) * 0.99)  # 99% for training, 1% for evaluation
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
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.use_amp)
    print(f"The UNet has Parameters={sum(p.numel() for p in unet.parameters() if p.requires_grad)}")

    # train
    print(
        f'MRI images are of shape {trainset[0].shape}. Use {len(trainset)} for training, {len(validset)} for evaluation. Unet model is loaded from {args.unet_ckpt}.'
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
            imgs = imgs.to(device)
            with autocast(enabled=args.use_amp):
                noise = torch.randn_like(imgs).to(device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (imgs.shape[0],), device=imgs.device
                ).long()
                target = noise
                # forward
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
        # valid
        if valid_loader is not None and epoch + 1 == args.n_epochs: # (step + 1) % 2000 == 0 or (step + 1) == len(train_loader):
            v_loss = 0
            for valid_step, valid_imgs in valid_bar:
                unet.eval()
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
                v_loss += valid_loss.item()
                valid_bar.set_postfix({'loss': v_loss / (valid_step + 1)})
        # save checkpoints
        torch.save(unet.state_dict(), os.path.join(args.output_dir, f'ddpm-ep-{epoch + 1}.pth'))
    # save the last checkpoint
    torch.save(unet.state_dict(), os.path.join(args.output_dir, 'ddpm-last.pth'))

