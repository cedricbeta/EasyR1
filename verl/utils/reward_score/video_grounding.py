import re

def video_grounding(predict_str, ground_truth):
    """
    Enhanced reward function for video grounding that evaluates:
    1. Format compliance (thinking and grounding tags)
    2. Accuracy of layer and segment_id prediction
    3. Reasonableness of sampling rate prediction
    
    Args:
        predict_str: Model's prediction string
        ground_truth: Ground truth string with <grounding>layer, segment, 0</grounding>
            Note: The third parameter (sampling rate) in ground truth is a placeholder
            
    Returns:
        float: Score between 0 and 1
    """
    # 1. Format checking (30% of score)
    format_score = check_format(predict_str)
    
    # If format is completely wrong, return 0
    if format_score == 0:
        return 0.0
    
    # 2. Extract predictions and ground truth
    pred_match = re.search(r'<grounding>([^<]+)</grounding>', predict_str)
    gt_match = re.search(r'<grounding>([^<]+)</grounding>', ground_truth)
    
    if not pred_match or not gt_match:
        return 0.1 * format_score  # Small score for partial format compliance
    
    try:
        pred_parts = [p.strip() for p in pred_match.group(1).split(',')]
        gt_parts = [p.strip() for p in gt_match.group(1).split(',')]
        
        if len(pred_parts) != 3 or len(gt_parts) != 2:
            return 0.2 * format_score  # Partial score for almost correct format
        
        # 3. Layer and segment accuracy (50% of score)
        grounding_score = evaluate_grounding(pred_parts, gt_parts)
        
        # 4. Sampling rate reasonableness (20% of score)
        sampling_score = evaluate_sampling_rate(pred_parts[2])
        
        # Final weighted score
        final_score = 0.3 * format_score + 0.5 * grounding_score + 0.2 * sampling_score
        return final_score
        
    except Exception as e:
        # If there's an error parsing the values, give a small score for format
        return 0.1 * format_score


def check_format(response_str):
    """
    Check if the response follows the required format.
    Returns a score between 0 and 1.
    """
    has_thinking = '<think>' in response_str and '</think>' in response_str
    has_grounding = '<grounding>' in response_str and '</grounding>' in response_str
    
    # Check thinking tag content
    thinking_quality = 0.0
    thinking_match = re.search(r'<think>(.*?)</think>', response_str, re.DOTALL)
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        # Simple heuristic: longer thinking is better (up to a point)
        thinking_quality = min(1.0, len(thinking_content) / 200)
    
    if has_thinking and has_grounding:
        return 0.6 + 0.4 * thinking_quality  # Up to 1.0 depending on thinking quality
    elif has_thinking:
        return 0.3  # Only thinking tag
    elif has_grounding:
        return 0.5  # Only grounding tag
    else:
        return 0.0  # No required tags


def evaluate_grounding(pred_parts, gt_parts):
    """
    Evaluate the accuracy of layer and segment_id prediction.
    Returns a score between 0 and 1.
    """
    try:
        pred_layer = int(pred_parts[0])
        pred_segment = int(pred_parts[1])
        gt_layer = int(gt_parts[0])
        gt_segment = int(gt_parts[1])
        
        # Exact match
        if pred_layer == gt_layer and pred_segment == gt_segment:
            return 1.0
        
        # Same layer, different segment
        if pred_layer == gt_layer:
            # Calculate how far off the segment is
            num_segments = 2**gt_layer
            segment_distance = min(
                abs(pred_segment - gt_segment),  # Direct distance
                num_segments - abs(pred_segment - gt_segment)  # Wrap-around distance (for circular videos)
            )
            segment_score = max(0, 1 - segment_distance / (num_segments/2))
            return 0.8 * segment_score
        
        # Different layer
        layer_diff = abs(pred_layer - gt_layer)
        layer_score = max(0, 1 - layer_diff/4)  # Assuming max layer difference is 4
        
        # If layers are different, segments can't be directly compared
        # But we can give partial credit
        return 0.4 * layer_score
    
    except (ValueError, IndexError):
        return 0.0  # Invalid format


def evaluate_sampling_rate(sampling_rate_str):
    """
    Evaluate if the predicted sampling rate is reasonable.
    Returns a score between 0 and 1.
    """
    try:
        rate = int(sampling_rate_str)
        
        # Sampling rate should be positive and not too large
        if rate <= 0:
            return 0.0
        elif rate <= 5:
            return 1.0  # Common reasonable rates (1-5 fps)
        elif rate <= 10:
            return 0.8  # Higher but still reasonable
        elif rate <= 30:
            return 0.5  # Very high but technically possible
        else:
            return 0.2  # Unreasonably high
    
    except ValueError:
        return 0.0  # Not a valid integer