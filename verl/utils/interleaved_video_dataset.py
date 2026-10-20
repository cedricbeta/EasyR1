# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from io import BytesIO
from typing import Any, Optional, Union, List, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import cv2

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def sample_video_frames(video_path: str, target_frames: int = 64) -> tuple[List[ImageObject], List[float]]:
    """
    Sample frames evenly from the entire video with explicit timestamps.

    Args:
        video_path: Path to the video file
        target_frames: Number of frames to extract

    Returns:
        Tuple of (List of PIL Images, List of timestamps in seconds)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or fps <= 0:
        print(f"Error: Invalid video properties - frames: {total_frames}, fps: {fps}")
        cap.release()
        return [], []

    # Sample frames evenly
    if total_frames <= target_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    frames = []
    timestamps = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            # Calculate timestamp for this frame
            timestamp = frame_idx / fps if fps > 0 else 0.0
            timestamps.append(timestamp)

    cap.release()
    return frames, timestamps


class InterleavedVideoDataset(Dataset):
    """
    Dataset for interleaved two-stage video question answering.
    Each sample contains the original question for stage 1 time span localization.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        video_key: str = "videos",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.video_key = video_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Load dataset and group by question pairs
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Group stage 1 and stage 2 data by video_id and question_id
        stage1_data = {}
        stage2_data = {}

        for item in data:
            metadata = item.get('metadata', {})
            video_id = metadata.get('video_id')
            question_id = metadata.get('question_id', '').replace('_stage1', '').replace('_stage2', '')
            stage = metadata.get('stage')

            if stage == 1:
                stage1_data[f"{video_id}_{question_id}"] = item
            elif stage == 2:
                stage2_data[f"{video_id}_{question_id}"] = item

        # Create pairs for interleaved training
        self.question_pairs = []
        for key in stage1_data:
            if key in stage2_data:
                stage1_item = stage1_data[key]
                stage2_item = stage2_data[key]

                # Extract original question from stage 1
                stage1_conv = stage1_item['conversations'][0]['value']
                original_question = stage1_conv.split('\n\nFirst, identify the time span')[0]

                # Create stage 1 prompt for time span localization with timestamps
                stage1_prompt = f"{original_question}\n\nFirst, identify the time span in the video where the relevant information can be found to answer this question. Respond in JSON format."
                stage1_answer = stage1_item['conversations'][1]['value']

                # Store both stages
                pair = {
                    'video_path': stage1_item['video'],
                    'original_question': original_question,
                    'stage1_prompt': stage1_prompt,
                    'stage1_answer': stage1_answer,
                    'stage2_item': stage2_item,
                    'metadata': stage1_item['metadata']
                }
                self.question_pairs.append(pair)

        print(f"Created {len(self.question_pairs)} question pairs for interleaved training")

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            # Filter based on stage 1 prompts (which are typically longer)
            original_len = len(self.question_pairs)
            self.question_pairs = [pair for pair in self.question_pairs if self._filter_overlong_prompts(pair)]
            print(f"Filtered {original_len - len(self.question_pairs)} overlong prompts")

    def _build_messages(self, prompt_str: str) -> list[dict[str, Any]]:
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        # Process video content
        content_list = []
        for i, content in enumerate(prompt_str.split("<video>")):
            if i != 0:
                content_list.append({"type": "video"})

            if content:
                content_list.append({"type": "text", "text": content})

        return [{"role": "user", "content": content_list}]

    def _filter_overlong_prompts(self, pair: dict[str, Any]) -> bool:
        # Check stage 1 prompt length (typically longer due to full video)
        try:
            # Get frames and timestamps
            processed_video, timestamps = sample_video_frames(pair['video_path'], target_frames=64)

            # Build messages with individual frames and timestamps
            content_list = []
            for i, (frame, timestamp) in enumerate(zip(processed_video, timestamps)):
                content_list.append({"type": "image"})
                content_list.append({"type": "text", "text": f" @{timestamp:.2f}s"})

            content_list.append({"type": "text", "text": f"\n\n{pair['stage1_prompt']}"})
            messages = [{"role": "user", "content": content_list}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            model_inputs = self.processor(
                images=processed_video, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        except:
            return False

    def __len__(self):
        return len(self.question_pairs)

    def __getitem__(self, index):
        pair = self.question_pairs[index]

        # For training, we only return stage 1 data
        # Stage 2 will be handled in the rollout/reward computation

        # Debug print for first few samples
        if index < 3:
            print(f"\n=== DEBUG SAMPLE {index} ===")
            print(f"Video path: {pair['video_path']}")
            print(f"Original question: {pair['original_question'][:100]}...")
            print(f"Stage 1 prompt: {pair['stage1_prompt'][:100]}...")
            print(f"Stage 1 answer: {pair['stage1_answer'][:100]}...")

        # Use 32 frames for stage 1 (sufficient context for time span localization)
        processed_video, timestamps = sample_video_frames(pair['video_path'], target_frames=32)

        # Check if video loading failed
        if len(processed_video) == 0:
            print(f"Warning: No frames loaded from {pair['video_path']}, trying next sample...")
            # Try to find a working video by recursively trying next samples
            return self.__getitem__((index + 1) % len(self.question_pairs))

        if index < 3:
            print(f"Loaded {len(processed_video)} frames, timestamps: {timestamps[:5]}...")

        # Build messages with individual frames and timestamps
        content_list = []
        for i, (frame, timestamp) in enumerate(zip(processed_video, timestamps)):
            content_list.append({"type": "image"})
            content_list.append({"type": "text", "text": f" @{timestamp:.2f}s"})

        content_list.append({"type": "text", "text": f"\n\n{pair['stage1_prompt']}"})
        messages = [{"role": "user", "content": content_list}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if index < 3:
            print(f"Generated prompt length: {len(prompt)}")
            print(f"Prompt preview: {prompt[:200]}...")

        model_inputs = self.processor(
            images=processed_video, text=[prompt], add_special_tokens=False, return_tensors="pt"
        )

        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        # Create concatenated ground truth for training
        stage2_answer = pair['stage2_item']['conversations'][1]['value'] if len(pair['stage2_item'].get('conversations', [])) > 1 else ""
        concatenated_ground_truth = pair['stage1_answer'] + "\n\n" + stage2_answer

        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "ground_truth": concatenated_ground_truth,  # Concatenated for sequence training
            "multi_modal_data": {"images": processed_video},

            # Store individual ground truths for reward computation
            "stage1_ground_truth": pair['stage1_answer'],
            "stage2_ground_truth": stage2_answer,

            # Store additional data for stage 2
            "video_path": pair['video_path'],
            "original_question": pair['original_question'],
            "stage2_item": pair['stage2_item'],
            "metadata": pair['metadata']
        }

        return example