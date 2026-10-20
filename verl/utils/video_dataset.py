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


def clip_video_by_timespan(video_path: str, start_time: float, end_time: float, target_frames: int = 32) -> List[ImageObject]:
    """
    Extract frames from a video within a specific time span.

    Args:
        video_path: Path to the video file
        start_time: Start time in seconds
        end_time: End time in seconds
        target_frames: Number of frames to extract

    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    duration_frames = end_frame - start_frame

    # Sample frames evenly within the time span
    if duration_frames <= target_frames:
        frame_indices = list(range(start_frame, end_frame))
    else:
        frame_indices = np.linspace(start_frame, end_frame - 1, target_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()
    return frames


def sample_video_frames(video_path: str, target_frames: int = 64) -> List[ImageObject]:
    """
    Sample frames evenly from the entire video.

    Args:
        video_path: Path to the video file
        target_frames: Number of frames to extract

    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly
    if total_frames <= target_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()
    return frames


class TwoStageVideoDataset(Dataset):
    """
    Dataset for two-stage video question answering:
    Stage 1: Identify time spans with full video (64 frames)
    Stage 2: Answer questions with clipped video (32 frames)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        stage: int = 1,  # 1 or 2
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
        self.stage = stage
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.video_key = video_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Load and filter dataset by stage
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Filter by training stage
        stage_data = [item for item in data if item.get('training_stage') == stage]

        # Convert to HuggingFace dataset format
        formatted_data = []
        for item in stage_data:
            conversations = item['conversations']

            # Extract prompt from human message
            human_msg = conversations[0]['value']
            prompt = human_msg.replace('<video>', '<video>')  # Ensure consistent token

            # Extract answer from assistant message
            assistant_msg = conversations[1]['value']

            formatted_item = {
                self.prompt_key: prompt,
                self.answer_key: assistant_msg,
                self.video_key: [item['video']],  # Wrap in list for consistency
                'metadata': item.get('metadata', {}),
                'focus_timespan': item.get('focus_timespan', None)
            }
            formatted_data.append(formatted_item)

        self.dataset = HFDataset.from_list(formatted_data)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
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

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Process video based on stage
        video_path = example[self.video_key][0]

        if self.stage == 1:
            # Stage 1: Use full video with 64 frames
            processed_video = sample_video_frames(video_path, target_frames=64)
        else:
            # Stage 2: Use clipped video with 32 frames
            focus_timespan = example.get('focus_timespan')
            if focus_timespan and len(focus_timespan) > 0:
                start_time, end_time = focus_timespan[0]
                processed_video = clip_video_by_timespan(video_path, start_time, end_time, target_frames=32)
            else:
                # Fallback to full video if no timespan provided
                processed_video = sample_video_frames(video_path, target_frames=32)

        model_inputs = self.processor(
            videos=[processed_video], text=[prompt], add_special_tokens=False, return_tensors="pt"
        )
        return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        video_path = example.pop(self.video_key)[0]

        # Process video based on stage
        if self.stage == 1:
            # Stage 1: Use full video with 64 frames
            processed_video = sample_video_frames(video_path, target_frames=64)
        else:
            # Stage 2: Use clipped video with 32 frames
            focus_timespan = example.get('focus_timespan')
            if focus_timespan and len(focus_timespan) > 0:
                start_time, end_time = focus_timespan[0]
                processed_video = clip_video_by_timespan(video_path, start_time, end_time, target_frames=32)
            else:
                # Fallback to full video if no timespan provided
                processed_video = sample_video_frames(video_path, target_frames=32)

        video_fps_list = [2.0]  # Default FPS for processed frames
        model_inputs = self.processor(
            videos=[processed_video], text=[prompt], add_special_tokens=False, return_tensors="pt"
        )

        if "second_per_grid_ts" in self.processor.model_input_names:
            model_inputs["second_per_grid_ts"] = [2.0 / fps for fps in video_fps_list]

        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        example["multi_modal_data"] = {"videos": [video_path]}

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

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example