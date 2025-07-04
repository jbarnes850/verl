#!/bin/bash
set -x

# Task Classification Training Script
# Trains Qwen2.5-VL-3B model for binary task classification using GRPO

ENGINE=${1:-vllm}
CONFIG_FILE=${2:-config/task_classifier_grpo.yaml}

# Environment setup
export USE_OPTIMIZED_MODEL=0  # Disable optimized models for training stability
export CUDA_VISIBLE_DEVICES=0,1  # Use 2 A100s (80GB each)

# Data paths
DATA_DIR=${DATA_DIR:-"test_data_quick"}
TRAIN_DATA="$DATA_DIR/real_train.parquet"
VAL_DATA="$DATA_DIR/real_val.parquet"

# Check if data exists, if not generate it
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo "Training data not found. Generating test data..."
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: HF_TOKEN environment variable not set"
        echo "Please run: export HF_TOKEN=your_huggingface_token"
        exit 1
    fi
    python prepare_task_data.py --num-samples 10 --output-dir "$DATA_DIR"
    
    # Re-check if data was generated
    if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
        echo "Failed to generate training data"
        exit 1
    fi
fi

# Create output directory
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/task_classification_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

echo "Starting task classification training..."
echo "Training data: $TRAIN_DATA"
echo "Validation data: $VAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Engine: $ENGINE"

# Run baseline evaluation first (optional - skip if it fails)
echo ""
echo "=== RUNNING ZERO-SHOT BASELINE ==="
echo "This establishes performance before GRPO training..."
if python3 run_baseline.py --data-path "$VAL_DATA"; then
    echo "=== BASELINE COMPLETE ==="
else
    echo "=== BASELINE FAILED - CONTINUING WITH TRAINING ==="
fi
echo ""

# Launch training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=16 \
    data.return_raw_chat=True \
    data.image_key=images \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.reward_fn_key=ground_truth \
    data.custom_cls.path=verl.utils.dataset.task_classification_dataset \
    data.custom_cls.name=TaskClassificationDataset \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.strategy=fsdp \
    critic.optim.lr=0.0 \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    \
    reward_model.enable=false \
    reward_model.reward_manager=naive \
    \
    custom_reward_function.path=$(pwd)/task_classification_reward.py \
    custom_reward_function.name=task_classification_compute_reward \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_task_classification' \
    trainer.experiment_name="qwen2_5_vl_3b_binary_classifier_$(date +%Y%m%d_%H%M%S)" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 \
    hydra.run.dir="$OUTPUT_DIR" \
    $@

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"

# Run comprehensive evaluation
echo ""
echo "=== RUNNING COMPREHENSIVE EVALUATION ==="
if [ -f "$DATA_DIR/baseline_results.json" ]; then
    python3 evaluate_comprehensive.py \
        --model-path "$OUTPUT_DIR/final_model" \
        --data-path "$VAL_DATA" \
        --baseline-results "$DATA_DIR/baseline_results.json" \
        --output-dir "$OUTPUT_DIR/evaluation"
else
    echo "Warning: No baseline results found, running without comparison"
    python3 evaluate_comprehensive.py \
        --model-path "$OUTPUT_DIR/final_model" \
        --data-path "$VAL_DATA" \
        --output-dir "$OUTPUT_DIR/evaluation"
fi
echo "=== EVALUATION COMPLETE ==="#

echo "Task classification training pipeline completed!"
echo "Check outputs in: $OUTPUT_DIR"