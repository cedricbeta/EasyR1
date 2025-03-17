#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

# Configurable parameters
MODEL_PATH=${1:-"Qwen/Qwen2.5-VL-3B-Instruct"}  # First argument or default
DATASET_PATH=${2:-"/home/chendong/video-rl/EasyR1/data"}  # Second argument or default
N_GPUS=${3:-2}  # Third argument or default

SYSTEM_PROMPT="""You are a video analysis assistant that helps users find the most relevant parts of a video.
Given a question and a series of video frames, analyze the frames carefully to determine the most relevant segment that answers the question.

You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
The reasoning process MUST BE enclosed within <think> </think> tags.

Your answer must specify:
- layer: determines the granularity of segmentation (layer 0 = whole video, layer 1 = video split into 2 segments, layer 2 = 4 segments, etc.)
- segment id: which specific segment (starting from 0) contains the answer
- sampling rate: how many frames per second you recommend for optimal viewing of this segment

The final answer MUST BE enclosed within <grounding>layer, segment id, sampling rate</grounding> tags.

For example:
<think>
Looking at the frames, I can see that the relevant action happens in the middle portion of the video. 
The frames showing the key event appear to be around frames 10-15.
Given that the video is about 60 seconds long, this would place the relevant content in the second quarter.
Layer 2 would divide the video into 4 parts, and segment 1 (the second segment) would contain this action.
Since the action is somewhat fast, I would recommend a sampling rate of 4 frames per second to capture it well.
</think>
<grounding>2, 1, 4</grounding>
"""

# Run training
python -m verl.trainer.main \
    config=examples/vg_example.yaml \
    data.train_files=${DATASET_PATH}@train \
    data.val_files=${DATASET_PATH}@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=8 \
    trainer.n_gpus_per_node=${N_GPUS}