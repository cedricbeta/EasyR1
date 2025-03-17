echo "Creating datasets..."
DATASET_DIR="data/video_grounding"
mkdir -p "$DATASET_DIR"

# Step 3: Run the training
echo "Starting training..."
# Environment variables for vLLM
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

# Configurable parameters (can be passed as arguments)
MODEL_PATH=${1:-"Qwen/Qwen2.5-VL-3B-Instruct"}  # First argument or default
N_GPUS=${2:-2}  # Second argument or default

python -m verl.trainer.main \
  config="$CONFIG_FILE" \
  data.train_files="$DATASET_DIR/train" \
  data.val_files="$DATASET_DIR/test" \
  worker.actor.model.model_path="$MODEL_PATH" \
  trainer.n_gpus_per_node="$N_GPUS"

# Clean up
rm "$CONFIG_FILE"
echo "Training complete!"