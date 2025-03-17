import os
import json
import argparse
import random
from PIL import Image
import cv2
import numpy as np
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from tqdm import tqdm
import pandas as pd

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
    max_layer = 4
    best_layer = 0
    best_segment = 0
    best_iou = 0
    for layer in range(max_layer + 1):
        num_segments = 2**layer
        segment_duration = video_duration_sec / num_segments
        for segment_id in range(num_segments):
            segment_start = segment_id * segment_duration
            segment_end = (segment_id + 1) * segment_duration
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

def sample_frames_from_video(video_path, frames_per_min=1):
    """Extract frames from video and return them as a list of PIL images along with the video duration in minutes."""
    frames = []
    if not os.path.exists(video_path):
        print(f"Warning: Video file {video_path} does not exist.")
        return frames, 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return frames, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    duration_min = duration_sec / 60
    frame_interval = int(fps * 60 / frames_per_min)
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        frames.append(image)
        frame_idx += frame_interval
    cap.release()
    return frames, duration_min

def generate_video_grounding_data(data_path: str, video_dir: str, frames_per_min: int = 1):
    """
    Generator that yields examples for a video grounding dataset.
    It loops over all JSON files in data_path, and for each file:
      - Loads the JSON data (containing the video key and QA pairs)
      - Finds the corresponding video file in video_dir (using common video extensions)
      - Samples frames from the video (or generates fake frames if not found)
      - Processes each QA pair (parsing the time reference and computing the segment)
      - Yields a dictionary with keys:
          "images": list of PIL images,
          "problem": question prefixed with "<image>\n",
          "answer": the grounding string,
          "id": unique identifier,
          "time_reference": time reference string,
          "video_id": video key,
          "duration_min": video duration in minutes,
          "multiple_choice_answer": original answer,
          "ground_truth": same as answer
    """
    for file_name in os.listdir(data_path):
        if not file_name.endswith(".json"):
            continue
        json_file = os.path.join(data_path, file_name)
        with open(json_file, 'r') as f:
            data = json.load(f)
        video_key = data.get("key", os.path.splitext(file_name)[0])
        print(f"Processing video: {video_key}")
        # Look for video file using common extensions
        video_path = None
        for ext in [".mp4", ".webm", ".avi", ".mov"]:
            potential_path = os.path.join(video_dir, f"{video_key}{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        if not video_path:
            fake_duration = 60  # in minutes
            print(f"Video file for {video_key} not found. Using fake duration of {fake_duration} minutes.")
            frames = []
            for i in range(10):  # generate 10 fake frames
                img = Image.new('RGB', (320, 240), color=(i*20, 100, 200))
                frames.append(img)
            duration_min = fake_duration
        else:
            frames, duration_min = sample_frames_from_video(video_path, frames_per_min)
        print(f"Extracted {len(frames)} frames from video with duration {duration_min:.2f} minutes")
        qa_pairs = data.get("qa", [])
        for qa in qa_pairs:
            uid = qa.get("uid", os.path.splitext(file_name)[0])
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            time_reference = qa.get("time_reference", "")
            start_time, end_time = parse_time_reference(time_reference)
            layer, segment_id = compute_layer_segment(start_time, end_time, duration_min)
            ground_truth = f"<grounding>{layer}, {segment_id}, 0</grounding>"
            yield {
                "images": frames,
                "problem": "<image>\n" + question,
                "answer": ground_truth,
                "id": uid,
                "time_reference": time_reference,
                "video_id": video_key,
                "duration_min": float(duration_min),
                "multiple_choice_answer": answer,
                "ground_truth": ground_truth
            }

def main():
    parser = argparse.ArgumentParser(description="Create video grounding dataset and save as Parquet splits")
    parser.add_argument("--data_path", type=str, default="video_grounding", help="Directory containing JSON files (e.g., train)")
    parser.add_argument("--video_dir", type=str, default="video_grounding", help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, default="data/video_grounding", help="Output directory for saving Parquet files")
    parser.add_argument("--frames_per_min", type=int, default=1, help="Number of frames to extract per minute")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset from the generator
    dataset = Dataset.from_generator(
        generate_video_grounding_data,
        gen_kwargs={
            "data_path": args.data_path,
            "video_dir": args.video_dir,
            "frames_per_min": args.frames_per_min
        }
    )
    
    # Split the dataset 80:20 into train and test splits
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Cast the "images" column to a Sequence of ImageData objects for each split
    for split in split_dataset.keys():
        split_dataset[split] = split_dataset[split].cast_column("images", Sequence(ImageData()))
    
    # Convert each split to a pandas DataFrame and save as Parquet
    for split, ds in split_dataset.items():
        df = ds.to_pandas()
        parquet_path = os.path.join(args.output_dir, f"{split}.parquet")
        df.to_parquet(parquet_path)
        print(f"Saved {split} split as Parquet to {parquet_path}")

if __name__ == "__main__":
    main()