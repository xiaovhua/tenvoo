export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0

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
ddpm_steps=1000
unet_ckpt=./path/to/pretrained/ddpm_unet.pth

###################################### peft settings ######################################
# # select from (ff, lora or locon, lokr, loha, tenvoo-l, tenvoo-q)
ft_mode=tenvoo-l
rank=4
# # select from (conv1, conv2, skip, ds, us, to_q, to_k, to_v, attn_proj, time_proj, time_emb) for monai diffusion model
target_modules=conv1,conv2,to_q,to_v,time_proj,time_emb
joint=0
# # select from (sum_opposite_freeze_one, last_layer_zero)
initialize_mode=sum_opposite_freeze_one

###################################### training settings ######################################
n_epochs=10
warmup=1
accu_steps=4
batch_size=1
lr=5e-5
loss_mode=mse


python ft_ddpm.py \
  --dataset_name $dataset_name \
  --dataset_root $dataset \
  --mri_resolution $resolution \
  --scale_ratio $scale \
  --output_dir $output \
  --unet_ckpt $unet_ckpt \
  -r $rank \
  --ft_mode=$ft_mode \
  --target_modules $target_modules \
  --joint $joint \
  --initialize_mode $initialize_mode \
  --ddpm_steps $ddpm_steps \
  --accu_steps $accu_steps \
  --n_epochs $n_epochs \
  --warmup $warmup \
  --batch_size $batch_size \
  --lr $lr \
  --loss_mode $loss_mode



