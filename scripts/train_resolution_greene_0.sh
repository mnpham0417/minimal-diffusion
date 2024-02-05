#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:a100:2
#SBATCH --time=47:59:00
#SBATCH --mem=120GB
#SBATCH --job-name=mnist_resolution
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
        CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 8102 resolution.py --sampling-only --class-cond --batch-size 512 --sampling-steps 400 \
        --dataset mnist \
        --arch UNet_8 \
        --pretrained-ckpt /vast/mp5847/minimal-diffusion/trained_models_UNet_8_unit_sphere/UNet_8_mnist-epoch_100-timesteps_1000-class_condn_True_ema_0.9995.pt \
        --target-class 0 \
        --output-file-name "resolution_mnist_0_UNet_8_unit_sphere_epochs=100_clower=1_cupper=5_rslower=1_rsupper=3_nimages=10.txt" \
        --complexity-lower 1 \
        --complexity-upper 5 \
        --random-seed-lower 1 \
        --random-seed-upper 3 \
        --d 8 --num-images 10'
