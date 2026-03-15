#!/bin/bash
# ========================================================================
# Slime Markdown Table RL Multi-Node Training Script
# ========================================================================

set -e

export MODEL_PATH=${MODEL_PATH:-"/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303/iter_0004600_hf"}
export TRAIN_DATA=${TRAIN_DATA:-"/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/markdown_table_rl_train.parquet"}
export SAVE_PATH=${SAVE_PATH:-"/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/markdown_table_rl"}

export ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
export ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
export COLOCATE=${COLOCATE:-"true"}

export TRAIN_BACKEND=${TRAIN_BACKEND:-"fsdp"}
export NUM_ROLLOUT=${NUM_ROLLOUT:-100}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}

export ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-"grpo"}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
export KL_LOSS_TYPE=${KL_LOSS_TYPE:-"low_var_kl"}
export KL_COEF=${KL_COEF:-0.00}
export ENTROPY_COEF=${ENTROPY_COEF:-0.01}
export EPS_CLIP=${EPS_CLIP:-0.2}
export EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-0.28}

export OPTIMIZER=${OPTIMIZER:-"adam"}
export LR=${LR:-"5.0e-7"}
export LR_DECAY_STYLE=${LR_DECAY_STYLE:-"cosine"}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
export ADAM_BETA1=${ADAM_BETA1:-0.9}
export ADAM_BETA2=${ADAM_BETA2:-0.98}

export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.6}

# Let slime resolve the reward from metadata.rm_type=markdown_table.
export CUSTOM_RM_PATH=${CUSTOM_RM_PATH:-""}

export USE_WANDB=${USE_WANDB:-"true"}
export WANDB_MODE=${WANDB_MODE:-"offline"}
export WANDB_PROJECT=${WANDB_PROJECT:-"internvl-markdown-table-grpo"}
export WANDB_GROUP=${WANDB_GROUP:-"internvl-markdown-table"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/run.sh"
