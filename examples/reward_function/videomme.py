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
from typing import Any, List, Dict


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON response from model output.

    Args:
        response: Raw model response string

    Returns:
        Parsed JSON dictionary, or empty dict if parsing fails
    """
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        pass

    return {}


def stage1_format_reward(response: str) -> float:
    """
    Check if stage 1 response has correct JSON format with time_spans and reason.

    Args:
        response: Model response string

    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    parsed = parse_json_response(response)

    # Check required fields
    if "time_spans" not in parsed or "reason" not in parsed:
        return 0.0

    # Check time_spans format (list of [start, end] pairs)
    time_spans = parsed["time_spans"]
    if not isinstance(time_spans, list) or len(time_spans) == 0:
        return 0.0

    for span in time_spans:
        if not isinstance(span, list) or len(span) != 2:
            return 0.0
        if not all(isinstance(x, (int, float)) for x in span):
            return 0.0
        if span[0] >= span[1]:  # start should be before end
            return 0.0

    # Check reason is string
    if not isinstance(parsed["reason"], str) or len(parsed["reason"].strip()) == 0:
        return 0.0

    return 1.0


def stage2_format_reward(response: str) -> float:
    """
    Check if stage 2 response has correct JSON format with answer and reason.

    Args:
        response: Model response string

    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    parsed = parse_json_response(response)

    # Check required fields (flexible - either correct_answer or answer)
    has_answer = "correct_answer" in parsed or "answer" in parsed
    has_reason = "reason" in parsed

    if not (has_answer and has_reason):
        return 0.0

    # Check reason is string
    if not isinstance(parsed["reason"], str) or len(parsed["reason"].strip()) == 0:
        return 0.0

    return 1.0


def stage1_accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Compute accuracy reward for stage 1 time span localization.

    Args:
        response: Model response string
        ground_truth: Ground truth JSON string

    Returns:
        IoU-based reward between predicted and ground truth time spans
    """
    try:
        pred_parsed = parse_json_response(response)
        gt_parsed = json.loads(ground_truth)

        if "time_spans" not in pred_parsed or "time_spans" not in gt_parsed:
            return 0.0

        pred_spans = pred_parsed["time_spans"]
        gt_spans = gt_parsed["time_spans"]

        if not pred_spans or not gt_spans:
            return 0.0

        # Compute IoU for each predicted span with best matching ground truth span
        max_ious = []
        for pred_span in pred_spans:
            if len(pred_span) != 2:
                continue

            best_iou = 0.0
            for gt_span in gt_spans:
                if len(gt_span) != 2:
                    continue

                # Compute IoU
                pred_start, pred_end = pred_span
                gt_start, gt_end = gt_span

                intersection_start = max(pred_start, gt_start)
                intersection_end = min(pred_end, gt_end)

                if intersection_start < intersection_end:
                    intersection = intersection_end - intersection_start
                    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
                    iou = intersection / union if union > 0 else 0.0
                    best_iou = max(best_iou, iou)

            max_ious.append(best_iou)

        # Return average of best IoUs
        return sum(max_ious) / len(max_ious) if max_ious else 0.0

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0.0


def stage2_accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Compute accuracy reward for stage 2 answer generation.

    Args:
        response: Model response string
        ground_truth: Ground truth JSON string

    Returns:
        1.0 if answer matches, 0.0 otherwise
    """
    try:
        pred_parsed = parse_json_response(response)
        gt_parsed = json.loads(ground_truth)

        # Extract predicted answer
        pred_answer = None
        if "correct_answer" in pred_parsed:
            pred_answer = pred_parsed["correct_answer"]
        elif "answer" in pred_parsed:
            pred_answer = pred_parsed["answer"]

        # Extract ground truth answer
        gt_answer = None
        if "correct_answer" in gt_parsed:
            gt_answer = gt_parsed["correct_answer"]
        elif "answer" in gt_parsed:
            gt_answer = gt_parsed["answer"]

        if pred_answer is None or gt_answer is None:
            return 0.0

        # Normalize answers for comparison
        pred_answer = str(pred_answer).strip().upper()
        gt_answer = str(gt_answer).strip().upper()

        return 1.0 if pred_answer == gt_answer else 0.0

    except (json.JSONDecodeError, KeyError, TypeError):
        return 0.0


def compute_score_stage1(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.3) -> List[Dict[str, float]]:
    """
    Compute rewards for stage 1 (time span localization).

    Args:
        reward_inputs: List of reward input dictionaries
        format_weight: Weight for format score

    Returns:
        List of score dictionaries
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for videomme stage 1 reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        format_score = stage1_format_reward(response)
        accuracy_score = stage1_accuracy_reward(response, ground_truth)

        scores.append({
            "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
            "format": format_score,
            "accuracy": accuracy_score,
        })

    return scores


def compute_score_stage2(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2) -> List[Dict[str, float]]:
    """
    Compute rewards for stage 2 (answer generation).

    Args:
        reward_inputs: List of reward input dictionaries
        format_weight: Weight for format score

    Returns:
        List of score dictionaries
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for videomme stage 2 reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        format_score = stage2_format_reward(response)
        accuracy_score = stage2_accuracy_reward(response, ground_truth)

        scores.append({
            "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
            "format": format_score,
            "accuracy": accuracy_score,
        })

    return scores