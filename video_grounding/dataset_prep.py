import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import random
import argparse


def create_video_grounding_dataset(videos_dir, annotations_file, output_dir, initial_frames_per_min=1, split="train"):
    """
    Create a video grounding dataset compatible with EasyR1.
    
    Note: initial_frames_per_min is just for initial sampling of frames.
    The model will predict the optimal sampling rate.
    """
    # Load annotations
    # with open(annotations_file, 'r') as f:
    #     data = json.load(f)
    data = []
    with open(annotations_file, 'r') as f:
        for line in f:
            # Remove any trailing whitespace and ensure the line is not empty
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    dataset_records = []
    
    for video_entry in tqdm(data):
        video_key = video_entry["key"]
        video_path = os.path.join(videos_dir, f"{video_key}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(videos_dir, f"{video_key}.webm")
            if not os.path.exists(video_path):
                print(f"Video for {video_key} not found, skipping...")
                continue
        
        # Get video metadata
        video_info = video_entry["video_info"]
        duration_min = video_info["duration_minutes"]
        fps = video_info["fps"]
        
        # Calculate the total frames to extract
        total_frames = int(duration_min * initial_frames_per_min)
        frame_interval = int(duration_min * 60 * fps / total_frames)
        
        # Extract frames
        sampled_frames = []
        frames_dir = os.path.join(output_dir, "frames", video_key)
        os.makedirs(frames_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            continue
            
        for i in range(total_frames):
            frame_idx = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                sampled_frames.append(frame_path)
            else:
                print(f"Failed to extract frame {i} from {video_path}")
        
        cap.release()
        
        # Create QA pairs
        for qa in video_entry["qa"]:
            question = qa["question"]
            answer = qa["answer"]  # This is the multiple-choice answer (A,B,C,D)
            time_reference = qa["time_reference"]
            
            # Process time reference for grounding
            start_time, end_time = parse_time_reference(time_reference)
            
            # For training data, compute the ground truth layer and segment
            layer, segment_id = compute_layer_segment(start_time, end_time, duration_min)
            
            # Store ground truth with placeholder for sampling rate (model will predict this)
            # We use a placeholder value for the sampling rate in the ground truth
            ground_truth = f"<grounding>{layer}, {segment_id}, 0</grounding>"
            
            dataset_records.append({
                "problem": question,
                "answer": ground_truth,
                "images": sampled_frames,
                "time_reference": time_reference,
                "video_id": video_key,
                "duration_min": float(duration_min),
                "fps": float(fps),
                "multiple_choice_answer": answer,  # Original multiple-choice answer
                "initial_frames_per_min": initial_frames_per_min  # Store the initial sampling rate
            })
    
    # Split into train/test if needed
    if split == "full":
        all_records = dataset_records
    else:
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(dataset_records)
        
        if split == "train":
            all_records = dataset_records[:int(0.8 * len(dataset_records))]
        else:  # test
            all_records = dataset_records[int(0.8 * len(dataset_records)):]
    
    # Save dataset
    dataset = Dataset.from_dict({
        k: [r[k] for r in all_records] for k in all_records[0].keys()
    })
    
    dataset.save_to_disk(os.path.join(output_dir, split))
    print(f"Created {split} dataset with {len(all_records)} examples")
    
    return dataset


def parse_time_reference(time_ref):
    """Parse time reference in format '00:15-00:19'"""
    if not time_ref or "-" not in time_ref:
        return 0, 0
    
    parts = time_ref.split("-")
    if len(parts) != 2:
        return 0, 0
        
    start, end = parts
    
    try:
        start_min, start_sec = map(int, start.split(":"))
        end_min, end_sec = map(int, end.split(":"))
        
        start_time = start_min * 60 + start_sec
        end_time = end_min * 60 + end_sec
        
        return start_time, end_time
    except ValueError:
        return 0, 0


def compute_layer_segment(start_time, end_time, video_duration_min):
    """
    Compute optimal layer and segment ID based on time reference.
    
    Layer 0: Whole video as one segment
    Layer 1: Video split into 2 segments
    Layer 2: Video split into 4 segments
    Layer 3: Video split into 8 segments
    Layer 4: Video split into 16 segments
    """
    video_duration_sec = video_duration_min * 60
    
    # Start with finest granularity we want to consider
    max_layer = 4
    best_layer = 0
    best_segment = 0
    best_iou = 0
    
    # Try each layer to find the best match
    for layer in range(max_layer + 1):
        num_segments = 2**layer
        segment_duration = video_duration_sec / num_segments
        
        # Find segment with maximum IoU
        for segment_id in range(num_segments):
            segment_start = segment_id * segment_duration
            segment_end = (segment_id + 1) * segment_duration
            
            # Calculate IoU
            intersection_start = max(start_time, segment_start)
            intersection_end = min(end_time, segment_end)
            
            if intersection_end > intersection_start:
                intersection = intersection_end - intersection_start
                union = max(end_time, segment_end) - min(start_time, segment_start)
                iou = intersection / union
                
                # Prefer more granular layers if IoU is at least 0.7
                if iou > best_iou or (iou >= 0.7 and layer > best_layer):
                    best_iou = iou
                    best_layer = layer
                    best_segment = segment_id
    
    return best_layer, best_segment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare video grounding dataset")
    parser.add_argument("--videos_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to annotations JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--initial_frames_per_min", type=int, default=1, help="Initial number of frames to sample per minute")
    args = parser.parse_args()
    
    # Create train split
    create_video_grounding_dataset(
        videos_dir=args.videos_dir,
        annotations_file=args.annotations_file,
        output_dir=args.output_dir,
        initial_frames_per_min=args.initial_frames_per_min,
        split="train"
    )
    
    # Create test split
    # create_video_grounding_dataset(
    #     videos_dir=args.videos_dir,
    #     annotations_file=args.annotations_file,
    #     output_dir=args.output_dir,
    #     initial_frames_per_min=args.initial_frames_per_min,
    #     split="test"
    # )