#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:mi50
#SBATCH --job-name=torch
#SBATCH --output=cascade_slurm_output/hudson_train_CIFAR10_cascaded_serial_mfc_lambda_sweep_blur1.0.out

/share/apps/singularity/bin/singularity \
    exec --overlay /scratch/sbp354/singularity_files/AMD-overlay-10GB-400K.ext3:ro \
    /scratch/work/public/singularity/hudson/images/rocm4.2-ubuntu20.04.sif \
    /bin/bash -c "
source ext3/env_hudson.sh
bash ../CascadedNets/train.sh
"

