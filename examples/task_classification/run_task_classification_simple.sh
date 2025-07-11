#!/bin/bash
set -x

# Simple Task Classification Training Script
# Uses standard dataset loading without custom class

ENGINE=${1:-vllm}

# Environment setup
export USE_OPTIMIZED_MODEL=0
export CUDA_VISIBLE_DEVICES=0,1

# Data paths
DATA_DIR=${DATA_DIR:-"desktop_task_data_verl"}
TRAIN_DATA="$DATA_DIR/verl_train.parquet"
VAL_DATA="$DATA_DIR/verl_val.parquet"

# Check if VERL format data exists, if not convert it
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo "Converting data to VERL format..."
    python convert_to_verl_format.py \
        --input-dir desktop_task_data_balanced \
        --output-dir "$DATA_DIR"
    
    if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
        echo "ERROR: Data conversion failed"
        exit 1
    fi
fi

# Create output directory
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/task_classification_simple_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

echo "Starting simplified task classification training..."
echo "Training data: $TRAIN_DATA"
echo "Validation data: $VAL_DATA"
echo "Output directory: $OUTPUT_DIR"

# Launch training without custom dataset class
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=16 \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    critic.strategy=fsdp \
    critic.optim.lr=0.0 \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    \
    reward_model.enable=false \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_task_classification' \
    trainer.experiment_name="simple_classifier_$(date +%Y%m%d_%H%M%S)" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=5 \
    hydra.run.dir="$OUTPUT_DIR"

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"