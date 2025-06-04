export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=1


###################################### dataset settings ######################################
dataset_name=adni
dataset=/path/to/adni/
resolution="[(160,224,160)]"
scale=1

# dataset_name=brats
# dataset=/path/to/brats/
# resolution="[(192,192,144)]"
# scale=1.5

# dataset_name=ppmi
# dataset=/path/to/ppmi/
# resolution="[(160,224,160)]"
# scale=1

###################################### output settings ######################################
output=./results/

###################################### model settings ######################################
unet_ckpt=./path/to/pretrained/ddpm_unet.pth
peft_ckpt=./path/to/fine-tuned/peft.pth
med3d_ckpt=./path/to/med3d_resnet50.pth

###################################### peft settings ######################################
# # select from (ff, lora or locon, lokr, loha, tenvoo-l, tenvoo-q)
ft_mode=tenvoo-l
rank=4
# # select from (conv1, conv2, skip, ds, us, to_q, to_k, to_v, attn_proj, time_proj, time_emb) for monai diffusion model
target_modules=conv1,conv2,to_q,to_v,time_proj,time_emb

python eval.py \
  --dataset_name $dataset_name \
  --dataset_root $dataset \
  --mri_resolution $resolution \
  --scale_ratio $scale \
  --output_dir $output \
  --unet_ckpt $unet_ckpt \
  --peft_ckpt $peft_ckpt \
  --med3d_ckpt $med3d_ckpt \
  -r $rank \
  --ft_mode=$ft_mode \
  --target_modules $target_modules

