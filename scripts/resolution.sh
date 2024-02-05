#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 8102 resolution.py --sampling-only --class-cond --batch-size 512 --sampling-steps 400 \
        --dataset mnist \
        --arch UNet_4 \
        --pretrained-ckpt /vast/mp5847/minimal-diffusion/trained_models_UNet_4_unit_sphere_ft_0_lr=0.0001_epochs=100_resteer=0_to_rest/UNet_4_mnist-epoch_100-timesteps_1000-class_condn_True_ema_0.9995.pt \
        --target-class 0 \
        --output-file-name "resolution_mnist_0_UNet_4_unit_sphere_ft_0_lr=0.0001_epochs=100_resteer=0_to_rest_clower=1_cupper=10_rslower=1_rsupper=10_nimages=10.txt" \
        --complexity-lower 1 \
        --complexity-upper 10 \
        --random-seed-lower 1 \
        --random-seed-upper 10 \
        --d 4 --num-images 50