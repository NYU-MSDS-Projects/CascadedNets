#!/bin/bash

DATASET_ROOT="/scratch/work/public/imagenet"  # Specify location of datasets
EXPERIMENT_ROOT="../cascade_output/experiments"  # Specify experiment root
SPLIT_IDXS_ROOT="../cascade_output/split_idx"  # Specify root of dataset split_idxs

# Experiment name to evaluate
MODEL="resnet18"  # resnet18, resnet34, resnet50, densenet_cifar
DATASET_NAME="ImageNet2012_16classes"  # CIFAR10, CIFAR100, TinyImageNet, ImageNet2012
EXPERIMENT_NAME="${MODEL}_${DATASET_NAME}"

TRAIN_MODE="cascaded"  # baseline, cascaded_seq, cascaded
CASCADED_SCHEME="serial"  # parallel, serial (used for train_mode=cascaded_seq)
DATASET_KEY="val"  # used for train_mode=cascaded_seq
BATCH_SIZE=128

TDL_MODE="OSD"  # OSD, EWS, noise
TDL_ALPHA=0.9
NOISE_VAR=0.0  # Used for noise kernel only
N_TIMESTEPS=70  # Used for EWS kernel only

#Image perturbations
GRAYSCALE=false
GAUSS_NOISE=false
GAUSS_NOISE_STD=0.0
BLUR=false
BLUR_STD=0.0

DEVICE=0
KEEP_LOGITS=true
KEEP_EMBEDDINGS=false
FORCE_OVERWRITE=true
DEBUG=false

cmd=( python ../CascadedNets/eval.py )   # create array with one element
cmd+=( --device $DEVICE )
cmd+=( --dataset_root $DATASET_ROOT )
cmd+=( --dataset_name $DATASET_NAME )
cmd+=( --dataset_key $DATASET_KEY )
cmd+=( --split_idxs_root $SPLIT_IDXS_ROOT )
cmd+=( --experiment_root $EXPERIMENT_ROOT )
cmd+=( --experiment_name $EXPERIMENT_NAME )
cmd+=( --train_mode $TRAIN_MODE )
cmd+=( --batch_size $BATCH_SIZE )
cmd+=( --cascaded_scheme $CASCADED_SCHEME )
cmd+=( --tdl_mode $TDL_MODE )
cmd+=( --tdl_alpha $TDL_ALPHA )
cmd+=( --noise_var $NOISE_VAR )
cmd+=( --n_timesteps $N_TIMESTEPS )

${KEEP_LOGITS} && cmd+=( --keep_logits )
${KEEP_EMBEDDINGS} && cmd+=( --keep_embeddings )
${DEBUG} && cmd+=( --debug ) && echo "DEBUG MODE ENABLED"
${GRAYSCALE} && cmd+=( --grayscale ) #pg_grayscale
${GAUSS_NOISE} && cmd+=( --gauss_noise )
${GAUSS_NOISE_STD} && cmd+=( --gauss_noise_std )
${BLUR} && cmd+=( --blur )
${BLUR_STD} && cmd+=( --blur )
${FORCE_OVERWRITE} && cmd+=( --force_overwrite ) && echo "FORCE OVERWRITE"

# Run command
"${cmd[@]}"