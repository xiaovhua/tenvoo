export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0

###################################### dataset settings ######################################
dataset_name=adni
dataset=/path/to/adni/
# dataset_name=brats
# dataset=/path/to/brats/
# dataset_name=ppmi
# dataset=/path/to/ppmi/

###################################### output settings ######################################
output=./results/

###################################### model settings ######################################
ddpm_steps=1000
unet_ckpt=./path/to/pretrained/ddpm_unet

###################################### peft settings ######################################
# # select from (ff, lora, locon, lokr, loha, tenvoo-l, tenvoo-q)
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



