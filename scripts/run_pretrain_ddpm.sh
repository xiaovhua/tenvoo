export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0

###################################### dataset settings ######################################
dataset_name=ukb
dataset_root=/path/to/ukb/t1/dataset
resolution="[(160,224,160)]"
scale=1

###################################### output settings ######################################
output=./results/ddpm

###################################### model settings ######################################
# unet_ckpt=./path/to/initialized/ddpm/unet # (optional)
ddpm_steps=1000

###################################### training settings ######################################
n_epochs=10
warmup=1
accu_steps=4
batch_size=1
lr=5e-5
loss_mode=mse


python pretrain_ddpm.py \
  --dataset_name $dataset_name \
  --dataset_root $dataset \
  --mri_resolution $resolution \
  --scale_ratio $scale \
  --output_dir $output \
  --ddpm_steps $ddpm_steps \
  --accu_steps $accu_steps \
  --n_epochs $n_epochs \
  --warmup $warmup \
  --batch_size $batch_size \
  --lr $lr \
  --loss_mode $loss_mode
