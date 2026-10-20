#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_stage1.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=videomme_stage1_timespan_localization \
    trainer.n_gpus_per_node=8