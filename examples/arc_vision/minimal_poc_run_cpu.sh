#!/bin/bash
# Minimal POC for CPU-only testing (e.g., Apple Silicon)
# This is just for verifying the training loop works, not for actual training

set -ex

# Use your existing prepared data
TRAIN_DATA=${TRAIN_DATA:-"~/data/arc_vision/screenspot/train.parquet"}
VAL_DATA=${VAL_DATA:-"~/data/arc_vision/screenspot/validation.parquet"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/arc_vision_poc_cpu"}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Expand paths
TRAIN_DATA=$(eval echo $TRAIN_DATA)
VAL_DATA=$(eval echo $VAL_DATA)

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting minimal POC training (CPU mode)..."
echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""
echo "WARNING: This is CPU-only mode for testing. Actual training should use GPU!"
echo ""

# Set CPU-only environment
export CUDA_VISIBLE_DEVICES=""

# Launch minimal training - remove invalid config fields
python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.return_raw_chat=True \
    data.image_key=images \
    data.reward_fn_key=data_source \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.do_sample=False \
    \
    custom_reward_function.path="$SCRIPT_DIR/arc_vision_custom_reward_minimal.py" \
    custom_reward_function.name=arc_vision_compute_reward \
    \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=arc_vision_poc_cpu \
    trainer.experiment_name=minimal_test_cpu \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.total_training_steps=1 \
    trainer.device=cpu \
    $@