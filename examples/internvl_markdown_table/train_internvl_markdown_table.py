"""
InternVL Markdown Table RL 训练脚本

使用 SLIME 框架进行 GRPO 训练，reward 从 metadata.rm_type=markdown_table 自动解析。
"""
import os

import slime.utils.external_utils.command_utils as U

NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303/iter_0004600_hf",
)
TRAIN_DATA = os.environ.get(
    "TRAIN_DATA",
    "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/markdown_table_rl_train.parquet",
)
SAVE_PATH = os.environ.get(
    "SAVE_PATH",
    "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/markdown_table_rl_single",
)


def execute():
    ckpt_args = f"--hf-checkpoint {MODEL_PATH} "

    rollout_args = (
        f"--prompt-data {TRAIN_DATA} "
        "--input-key problem "
        "--label-key answer "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--num-rollout 100 "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.7 "
        "--global-batch-size 64 "
        "--rollout-stop-token-ids 151645 "
    )

    multimodal_args = '--multimodal-keys \'{"image":"images"}\' '

    fsdp_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.01 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 5e-7 "
        "--lr-decay-style cosine "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 500 "
        "--sglang-enable-metrics "
        "--attn-implementation flash_attention_2 "
        "--sglang-cuda-graph-max-bs 32 "
    )

    save_args = (
        f"--save {SAVE_PATH} "
        "--save-interval 10 "
    )

    wandb_project = os.environ.get("WANDB_PROJECT", "internvl-markdown-table-grpo")
    wandb_name = os.environ.get("WANDB_NAME", "internvl-markdown-table")
    wandb_team = os.environ.get("WANDB_ENTITY", "")
    wandb_host = os.environ.get("WANDB_BASE_URL", "")
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    wandb_args = f"--use-wandb --wandb-project {wandb_project} --wandb-group {wandb_name} "
    if wandb_team:
        wandb_args += f"--wandb-team {wandb_team} "
    if wandb_host:
        wandb_args += f"--wandb-host {wandb_host} "
    if wandb_key:
        wandb_args += f"--wandb-key {wandb_key} "

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{multimodal_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{fsdp_args} "
        f"{sglang_args} "
        f"{save_args} "
        f"{wandb_args} "
        f"{misc_args} "
    )

    extra_env_vars = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }

    for key in ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_NAME", "WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_MODE"]:
        if key in os.environ:
            extra_env_vars[key] = os.environ[key]

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    execute()
