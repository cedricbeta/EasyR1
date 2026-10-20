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
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import cv2
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers import PreTrainedTokenizer, ProcessorMixin

from .vllm_rollout_spmd import vLLMRollout
from ...utils.dataset import process_video


def clip_video_by_timespan(video_path: str, start_time: float, end_time: float, target_frames: int = 32) -> tuple[List[ImageObject], List[float]]:
    """
    Extract frames from a video within a specific time span with explicit timestamps.

    Args:
        video_path: Path to the video file
        start_time: Start time in seconds
        end_time: End time in seconds
        target_frames: Number of frames to extract

    Returns:
        Tuple of (List of PIL Images, List of timestamps in seconds)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    duration_frames = end_frame - start_frame

    # Sample frames evenly within the time span
    if duration_frames <= target_frames:
        frame_indices = list(range(start_frame, end_frame))
    else:
        frame_indices = np.linspace(start_frame, end_frame - 1, target_frames, dtype=int)

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
            timestamp = frame_idx / fps
            timestamps.append(timestamp)

    cap.release()
    return frames, timestamps


def parse_time_spans(response: str) -> List[List[float]]:
    """
    Parse time spans from stage 1 response.

    Args:
        response: Stage 1 model response

    Returns:
        List of [start_time, end_time] pairs
    """
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            if "time_spans" in parsed and isinstance(parsed["time_spans"], list):
                time_spans = []
                for span in parsed["time_spans"]:
                    if isinstance(span, list) and len(span) == 2:
                        start, end = span
                        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start < end:
                            time_spans.append([float(start), float(end)])
                return time_spans
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return []


class InterleavedVideoRollout(vLLMRollout):
    """
    Extended vLLM rollout for interleaved two-stage video QA.
    Performs stage 1, extracts time spans, then performs stage 2.
    """

    def __init__(
        self,
        model_path: str,
        config,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        super().__init__(model_path, config, tokenizer, processor)
        self.processor = processor

    def rollout(self, data: Any, **generation_kwargs) -> Dict[str, Any]:
        """
        Perform interleaved two-stage rollout:
        1. Stage 1: Generate time spans with full video (64 frames) - n=3
        2. Stage 2: Generate answers with clipped video as multi-images (32 frames) - n=1
        3. Concatenate responses for training with GRPO index consistency
        """
        # Extract batch information
        batch_size = len(data["input_ids"])
        device = data["input_ids"].device

        # STAGE 1 ROLLOUT: Time span localization only
        stage1_kwargs = generation_kwargs.copy()
        stage1_kwargs['n'] = 3  # Force n=3 for stage 1
        stage1_results = super().rollout(data, **stage1_kwargs)

        # STAGE 2 ROLLOUT: Answer generation with clipped video as multi-images
        stage1_responses = stage1_results["responses"]

        # Build stage 2 batch with new prompts and clipped images
        stage2_batch_data = []
        valid_stage2_indices = []
        stage2_failed_indices = []  # Track failed stage 2 for dummy responses

        for i, stage1_response in enumerate(stage1_responses):
            # Parse time spans from stage 1 response
            time_spans = parse_time_spans(stage1_response)

            if not time_spans:
                stage2_failed_indices.append(i)
                continue

            # Use the first time span
            start_time, end_time = time_spans[0]

            # Map back to original question index (since stage 1 has n=3 responses per question)
            original_question_idx = i // 3

            # Get stage 2 data from original batch
            stage2_item = data.get("stage2_item", [{}])[original_question_idx]
            if not stage2_item:
                stage2_failed_indices.append(i)
                continue

            # Get original question and stage 2 prompt
            original_question = data.get("original_question", [""])[original_question_idx]
            stage2_conv = stage2_item.get("conversations", [{}])

            if len(stage2_conv) > 0:
                video_path = data.get("video_path", [""])[original_question_idx]
                if video_path:
                    try:
                        # Clip video based on predicted time span and get timestamps
                        clipped_frames, timestamps = clip_video_by_timespan(video_path, start_time, end_time, target_frames=16)

                        # Create stage 2 prompt with multi-image format
                        stage2_prompt = f"{original_question}\n\nOptions:\n(A) Raises it and holds an open palm before lowering it.\n(B) Keeps it resting on the table the whole time.\n(C) Uses it to point toward the door behind them.\n(D) Waves a small object back and forth.\n\nFocus on the time spans [[{start_time}, {end_time}]] and provide the correct answer. Respond in JSON format."

                        # Build messages with individual frames and timestamps
                        content_list = []
                        for i, (frame, timestamp) in enumerate(zip(clipped_frames, timestamps)):
                            content_list.append({"type": "image"})
                            content_list.append({"type": "text", "text": f" @{timestamp:.2f}s"})

                        content_list.append({"type": "text", "text": f"\n\n{stage2_prompt}"})
                        messages = [{"role": "user", "content": content_list}]
                        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

                        # Process with multi-image format
                        model_inputs = self.processor(
                            images=clipped_frames, text=[prompt], add_special_tokens=False, return_tensors="pt"
                        )

                        stage2_batch_data.append({
                            "input_ids": model_inputs["input_ids"][0],
                            "attention_mask": model_inputs["attention_mask"][0],
                            "position_ids": torch.arange(model_inputs["input_ids"].size(-1)),
                            "multi_modal_data": {"images": clipped_frames}
                        })
                        valid_stage2_indices.append(i)

                    except Exception as e:
                        print(f"Error processing video for stage 2: {e}")
                        stage2_failed_indices.append(i)
                        continue
                else:
                    stage2_failed_indices.append(i)
            else:
                stage2_failed_indices.append(i)

        # Perform stage 2 rollout if we have valid samples
        stage2_results = {"responses": [], "log_probs": [], "prompt_log_probs": []}

        if len(stage2_batch_data) > 0:
            # Pad stage 2 batch
            max_len = max(item["input_ids"].size(-1) for item in stage2_batch_data)

            padded_input_ids = []
            padded_attention_mask = []
            padded_position_ids = []

            for item in stage2_batch_data:
                ids = item["input_ids"]
                mask = item["attention_mask"]
                pos_ids = item["position_ids"]

                pad_len = max_len - ids.size(-1)
                if pad_len > 0:
                    ids = torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id, device=ids.device), ids])
                    mask = torch.cat([torch.zeros(pad_len, device=mask.device), mask])
                    pos_ids = torch.cat([torch.zeros(pad_len, device=pos_ids.device), pos_ids])

                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
                padded_position_ids.append(pos_ids)

            stage2_data = {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "position_ids": torch.stack(padded_position_ids),
                "multi_modal_data": {"images": [item["multi_modal_data"]["images"] for item in stage2_batch_data]}
            }

            # Run stage 2 rollout with n=1
            stage2_kwargs = generation_kwargs.copy()
            stage2_kwargs['n'] = 1  # Force n=1 for stage 2
            stage2_results = super().rollout(stage2_data, **stage2_kwargs)

        # CONCATENATION: Create final training sequences with dummy responses for GRPO consistency
        concatenated_responses = []
        concatenated_log_probs = []
        concatenated_prompt_log_probs = []

        stage1_responses = stage1_results["responses"]
        stage2_responses = stage2_results.get("responses", [])
        stage1_log_probs = stage1_results["log_probs"]
        stage2_log_probs = stage2_results.get("log_probs", [])
        stage1_prompt_log_probs = stage1_results["prompt_log_probs"]

        # Map stage 2 responses back to stage 1 response indices
        stage2_response_map = {}
        stage2_log_prob_map = {}

        for idx, valid_stage1_idx in enumerate(valid_stage2_indices):
            if idx < len(stage2_responses):
                stage2_response_map[valid_stage1_idx] = stage2_responses[idx]
                if idx < len(stage2_log_probs):
                    stage2_log_prob_map[valid_stage1_idx] = stage2_log_probs[idx]

        # Create dummy stage 2 response for failed cases
        dummy_stage2_response = '{"error": "Failed to process time span", "answer": "(A)"}'

        # Build concatenated sequences as unified prompts with both visual contexts
        import torch
        concatenated_responses = []
        concatenated_log_probs = []
        concatenated_prompt_log_probs = []
        concatenated_multi_modal_data = []
        unified_prompts = []

        for i, stage1_response in enumerate(stage1_responses):
            # Get corresponding original question index (accounting for n=3 responses per question)
            original_question_idx = i // 3

            if i in stage2_response_map:
                stage2_response = stage2_response_map[i]

                # Get stage 2 visual data for this response
                stage2_idx = valid_stage2_indices.index(i)
                if stage2_idx < len(stage2_batch_data):
                    stage2_images = stage2_batch_data[stage2_idx]["multi_modal_data"]["images"]
                else:
                    stage2_images = []
            else:
                stage2_response = dummy_stage2_response
                stage2_images = []  # No stage 2 images for failed cases

            # Get original question and stage 1 images
            original_question = data.get("original_question", [""])[original_question_idx]
            stage1_images = data.get("multi_modal_data", {}).get("images", [])
            if isinstance(stage1_images, list) and len(stage1_images) > original_question_idx:
                stage1_frames = stage1_images[original_question_idx]
            else:
                stage1_frames = []

            # Build unified prompt that contains both stage 1 and stage 2 questions
            # This will be used for GRPO log probability computation
            content_list = []

            # Add all images with timestamps (stage1 + stage2 frames combined)
            combined_images = stage1_frames + stage2_images
            for j, frame in enumerate(combined_images):
                content_list.append({"type": "image"})
                # Calculate timestamp based on whether it's stage 1 or stage 2 frame
                if j < len(stage1_frames):
                    # Stage 1 frame timestamp (evenly distributed across full video)
                    timestamp = j * (64.36 if len(stage1_frames) > 1 else 0)
                else:
                    # Stage 2 frame timestamp (within clipped segment, offset by stage1 frames)
                    stage2_idx = j - len(stage1_frames)
                    timestamp = stage2_idx * (1.0 if len(stage2_images) > 1 else 0)
                content_list.append({"type": "text", "text": f" @{timestamp:.2f}s"})

            # Add unified question that includes both stage 1 and stage 2 tasks
            unified_question = f"""{original_question}

First, identify the time span in the video where the relevant information can be found to answer this question. Respond in JSON format with time_spans and reason.

Then, based on the identified time spans, answer the following multiple choice question:

Options:
(A) Raises it and holds an open palm before lowering it.
(B) Keeps it resting on the table the whole time.
(C) Uses it to point toward the door behind them.
(D) Waves a small object back and forth.

Provide the correct answer in JSON format with the answer and reason."""

            content_list.append({"type": "text", "text": f"\n\n{unified_question}"})

            # Build the unified prompt for GRPO
            messages = [{"role": "user", "content": content_list}]
            unified_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            # The "response" for GRPO is the concatenated stage1 + stage2 responses
            concatenated_response_text = stage1_response + "\n\n" + stage2_response
            concatenated_responses.append(concatenated_response_text)

            # Store combined images for multi-modal data
            concatenated_multi_modal_data.append({"images": combined_images})

            # Store the unified prompt for this sequence
            unified_prompts.append(unified_prompt)

            # For log probs, we'll recompute them during the training step
            concatenated_log_probs.append(None)
            concatenated_prompt_log_probs.append(stage1_prompt_log_probs[i] if i < len(stage1_prompt_log_probs) else None)

        # Return concatenated results for GRPO training
        # The concatenated sequences contain stage1_response + "\n\n" + stage2_response
        # Multi-modal data combines stage1 (32 frames) + stage2 (16 frames) = 48 total frames
        combined_results = {
            "responses": concatenated_responses,  # Use concatenated responses for GRPO
            "log_probs": concatenated_log_probs,
            "prompt_log_probs": concatenated_prompt_log_probs,
            "multi_modal_data": concatenated_multi_modal_data,  # Combined visual contexts

            # Store unified prompts for proper log probability computation
            "unified_prompts": unified_prompts,  # Store individual unified prompts for each sequence
            "is_unified_sequence": True,  # Flag to indicate this uses unified prompts

            # Store individual stage info for debugging/analysis
            "stage1_responses": stage1_results["responses"],
            "stage2_responses": stage2_results.get("responses", []),
            "valid_stage2_indices": valid_stage2_indices,
            "stage2_failed_indices": stage2_failed_indices,
            "parsed_time_spans": [parse_time_spans(resp) for resp in stage1_results["responses"]],
        }

        return combined_results

    def _build_video_messages(self, prompt_str: str) -> List[Dict[str, Any]]:
        """Build messages for video processing."""
        content_list = []
        for i, content in enumerate(prompt_str.split("<video>")):
            if i != 0:
                content_list.append({"type": "video"})
            if content:
                content_list.append({"type": "text", "text": content})

        return [{"role": "user", "content": content_list}]