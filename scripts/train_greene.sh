#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:a100:4
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
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8101 main.py --arch UNet --dataset mnist --class-cond --epochs 100 --batch-size 1024 --sampling-steps 100 '
