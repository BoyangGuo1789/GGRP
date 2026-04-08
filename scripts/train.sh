#!/bin/bash

set -euo pipefail

# Resolve repository root and run from there so relative config paths always work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Usage:
#   DATA_ROOT=/path/to/datasets GPU_ID=0 \
#   bash scripts/train.sh <DATASET> <NUM_CLASS> [TEACHER_MODE] [TEACHER_EMA_MOMENTUM] [PROMPT_LEN] [MAX_EPOCH]
#
# Example:
#   DATA_ROOT=/path/to/datasets GPU_ID=0 \
#   bash scripts/train.sh caltech101 100 ema 0.99 16 100

DATASET=${1:-}
NUM_CLASS=${2:-}
TEACHER_MODE=${3:-freeze}          # freeze or ema
TEACHER_EMA_MOMENTUM=${4:-0.99}    # used only when TEACHER_MODE=ema
PROMPT_LEN=${5:-16}
MAX_EPOCH=${6:-100}

if [ -z "${DATASET}" ] || [ -z "${NUM_CLASS}" ]; then
    echo "Usage: bash scripts/train.sh <DATASET> <NUM_CLASS> [TEACHER_MODE] [TEACHER_EMA_MOMENTUM] [PROMPT_LEN] [MAX_EPOCH]"
    exit 1
fi

if ! [[ "${NUM_CLASS}" =~ ^[0-9]+$ ]]; then
    echo "NUM_CLASS must be a positive integer."
    exit 1
fi

DATA_ROOT=${DATA_ROOT:-/path/to/datasets}
TRAINER=${TRAINER:-GGRP}
CFG=${CFG:-rn50}
CE_TARGET=${CE_TARGET:-teacher}     # teacher or noisy
SHOTS=${SHOTS:-16}
GPU_ID=${GPU_ID:-0}

# You can override the sweep dimensions via environment variables.
NOISE_TYPES_STR=${NOISE_TYPES:-"sym asym"}
NOISE_RATES_STR=${NOISE_RATES:-"0.125 0.25 0.375 0.50 0.625 0.75"}
# CoOp-style seed behavior:
# - seed >= 0: fixed seed
# - seed = -1: random mode (no fixed seed in train.py)
SEEDS_STR=${SEEDS:-"-1"}

IFS=' ' read -r -a NOISE_TYPES <<< "${NOISE_TYPES_STR}"
IFS=' ' read -r -a NOISE_RATES <<< "${NOISE_RATES_STR}"
IFS=' ' read -r -a SEED_LIST <<< "${SEEDS_STR}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Teacher setup:
# - freeze: static teacher (EMA=0.0)
# - ema:    update teacher with EMA momentum
if [ "${TEACHER_MODE}" = "ema" ]; then
    TEACHER_EMA=${TEACHER_EMA_MOMENTUM}
    TEACHER_TAG="ema_${TEACHER_EMA}"
else
    TEACHER_EMA=0.0
    TEACHER_TAG="freeze"
fi

for TYPE in "${NOISE_TYPES[@]}"; do
    for RATE in "${NOISE_RATES[@]}"; do
        for SEED in "${SEED_LIST[@]}"; do
            DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_ctx${PROMPT_LEN}_ep${MAX_EPOCH}/noise_${TYPE}_${RATE}/ce_${CE_TARGET}/teacher_${TEACHER_TAG}/seed${SEED}
            if [ -d "${DIR}" ]; then
                echo "Results already exist at ${DIR}. Skipping."
                continue
            fi

            echo "Running dataset=${DATASET}, num_class=${NUM_CLASS}, noise=${TYPE}/${RATE}, seed=${SEED}"
            python train.py \
                --root "${DATA_ROOT}" \
                --seed "${SEED}" \
                --trainer "${TRAINER}" \
                --dataset-config-file "configs/datasets/${DATASET}.yaml" \
                --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
                --output-dir "${DIR}" \
                DATASET.NUM_SHOTS "${SHOTS}" \
                DATASET.NOISE_RATE "${RATE}" \
                DATASET.NOISE_TYPE "${TYPE}" \
                DATASET.num_class "${NUM_CLASS}" \
                TRAINER.GGRP.CE_TARGET "${CE_TARGET}" \
                TRAINER.GGRP.TEACHER_EMA "${TEACHER_EMA}" \
                TRAINER.GGRP.N_CTX "${PROMPT_LEN}" \
                OPTIM.MAX_EPOCH "${MAX_EPOCH}" \
                DATALOADER.NUM_WORKERS 4
        done
    done
done
