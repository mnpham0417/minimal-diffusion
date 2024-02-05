#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:a100:2
#SBATCH --time=47:59:00
#SBATCH --mem=120GB
#SBATCH --job-name=miniman_diffusion_mnist
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=miniman_diffusion_mnist_%j.out

module purge

singularity exec --nv \
  --overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate /scratch/mp5847/conda_environments/conda_pkgs/diffusion_ft; cd /vast/mp5847/minimal-diffusion/; \
        CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 8101 main_ft.py \
        --arch UNet --dataset mnist --class-cond --epochs 70 --batch-size 512 --sampling-steps 100 --pretrained-ckpt "/vast/mp5847/minimal-diffusion/trained_models_unit_sphere/UNet_mnist-epoch_100-timesteps_1000-class_condn_True_ema_0.9995.pt" \
        --save-dir "./trained_models_unit_sphere_ft_0_lr=0.0001_epochs=70_resteer=0_to_1/" --lr 0.0001'
