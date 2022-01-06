#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=100:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --partition=v100
#SBATCH --job-name=torch
#SBATCH --output=cascade_slurm_output/train_16class_imagenet_cascaded_parallel_lambda_mfc_sweep_noise1.0.out

module purge

singularity exec --nv \
        --overlay /scratch/sbp354/singularity_files/pytorch1.8.0-cuda11.1.ext3:ro \
        /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
        /bin/bash -c "source ext3/env.sh; bash ../CascadedNets/train.sh"
