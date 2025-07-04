#!/bin/bash
set -x

# Task Classification Training Script
# Trains Qwen2.5-VL-3B model for binary task classification using GRPO

ENGINE=${1:-vllm}
CONFIG_FILE=${2:-config/task_classifier_grpo.yaml}

# Environment setup
export USE_OPTIMIZED_MODEL=0  # Disable optimized models for training stability
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use 4 GPUs for 3B model

# Data paths
DATA_DIR=${DATA_DIR:-"data"}
TRAIN_DATA="$DATA_DIR/train.parquet"
VAL_DATA="$DATA_DIR/val.parquet"

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found: $TRAIN_DATA"
    echo "Please run prepare_task_data.py first"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Validation data not found: $VAL_DATA"
    echo "Please run prepare_task_data.py first"
    exit 1
fi

# Create output directory
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/task_classification_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

echo "Starting task classification training..."
echo "Training data: $TRAIN_DATA"
echo "Validation data: $VAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Engine: $ENGINE"

# Launch training
python3 -m verl.trainer.main_ppo \
    --config-name task_classifier_grpo \
    --config-path $(pwd)/config \
    reward_fn=task_classification \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=16 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_task_classification' \
    trainer.experiment_name="qwen2_5_vl_3b_binary_classifier_$(date +%Y%m%d_%H%M%S)" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 \
    trainer.save_dir="$OUTPUT_DIR" \
    hydra.run.dir="$OUTPUT_DIR" \
    $@

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"

# Run evaluation
echo "Running evaluation..."
python3 evaluate_classifier.py --model-dir "$OUTPUT_DIR" --data-file "$VAL_DATA"

echo "Task classification training pipeline completed!"
echo "Check outputs in: $OUTPUT_DIR"