#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
STAGE1_CHECKPOINT_PATH=checkpoints/videomme_two_stage/videomme_stage1_timespan_localization/global_step_XXX/actor  # replace XXX with actual step

python3 -m verl.trainer.main \
    config=examples/config_stage2.yaml \
    worker.actor.model.model_path=${STAGE1_CHECKPOINT_PATH} \
    trainer.experiment_name=videomme_stage2_answer_generation \
    trainer.n_gpus_per_node=8