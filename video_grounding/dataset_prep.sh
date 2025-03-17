echo "Creating datasets..."
DATASET_DIR="data/video_grounding"
mkdir -p "$DATASET_DIR"

python video_grounding/create_video_grounding_dataset.py \
  --json_file video_grounding/data.json \
  --video_dir video_grounding \
  --output_dir "$DATASET_DIR" \
  --frames_per_min 1
