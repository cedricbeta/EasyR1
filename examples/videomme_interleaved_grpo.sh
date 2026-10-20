#!/bin/bash

set -x

# export WANDB_API_KEY=REDACTED_SET_VIA_ENV
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
# MODEL_PATH=/home/chendong/video-rl/qwen-sft/qwen-vl-sft/output_two_stage/checkpoint-182
MODEL_PATH=/workspace/video-rl/qwen-sft/qwen-vl-sft/output_two_stage/checkpoint-182

python3 -m verl.trainer.main \
    config=examples/config_interleaved.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=videomme_interleaved_two_stage_grpo \
    trainer.n_gpus_per_node=4