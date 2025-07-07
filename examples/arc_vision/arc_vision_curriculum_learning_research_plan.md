# Arc Vision Curriculum Learning Research Plan

## Research Objective

Evaluate the impact of curriculum learning on Arc Vision RL training by comparing three models:
1. **Baseline Model**: Original Qwen2.5-VL-3B-Instruct (no training)
2. **Standard GRPO Model**: Trained with standard GRPO without curriculum learning (up to step 50)
3. **Curriculum GRPO Model**: Trained with GRPO + curriculum learning (up to step 50)

## Research Questions

1. Does curriculum learning prevent the model from converging to tool-only responses?
2. How does curriculum learning affect the learning curve and convergence speed?
3. What is the impact on final performance metrics (IoU, tool usage, task completion)?
4. Does curriculum learning lead to more stable training dynamics?

## Experimental Setup

### Model Specifications
- **Base Model**: Qwen/Qwen2.5-VL-3B-Instruct
- **Training Algorithm**: GRPO (Generalized Reward Policy Optimization)
- **Dataset**: ScreenSpot (train.parquet, validation.parquet)
- **Training Steps**: 50 steps for both trained models
- **Batch Size**: 8 (safe configuration)
- **GPU Memory**: 0.4 utilization (to avoid CUDA errors)

### Current Status
- **Standard GRPO Model**: Already trained, checkpoint at step 25 available
  - Path: `/home/user/verl/outputs/arc_vision/global_step_25/`
  - Need to continue training to step 50
- **Curriculum GRPO Model**: To be trained from scratch

## Implementation Plan

### Phase 1: Complete Standard GRPO Training (Steps 26-50)

1. **Resume Training**
   ```bash
   # Modify run_arc_vision_grpo_safe.sh to resume from step 25
   trainer.resume_mode=resume_path
   trainer.resume_from_path="/home/user/verl/outputs/arc_vision/global_step_25"
   ```

2. **Monitor for Best Checkpoint**
   - Track validation IoU
   - Track entropy (should not drop below 0.2)
   - Save best checkpoint based on task completion rate

### Phase 2: Implement Curriculum Learning

1. **Dataset Preparation**
   ```python
   # Create curriculum stages
   Stage 1 (Steps 0-10): Large objects only (bbox_area > 0.05)
   Stage 2 (Steps 11-25): Medium + large objects (bbox_area > 0.03)
   Stage 3 (Steps 26-40): Balanced dataset (70% small, 30% large)
   Stage 4 (Steps 41-50): Full dataset with slight augmentation
   ```

2. **Reward Function Modifications**
   ```python
   def curriculum_aware_reward(step, is_small_object, has_tools, has_bbox):
       if step < 10:
           # Stage 1: Encourage bbox attempts
           if has_bbox:
               return base_reward * 1.5  # Bonus for bbox attempts
           else:
               return -0.1  # Penalty for not trying
       elif step < 25:
           # Stage 2: Balanced rewards
           return standard_reward
       else:
           # Stage 3-4: Decay tool rewards
           tool_decay = max(0.1, 1.0 - (step - 25) / 25)
           return standard_reward * tool_decay
   ```

3. **Training Script Modifications**
   - Create `run_arc_vision_grpo_curriculum.sh`
   - Add curriculum dataset loader
   - Implement stage-based reward function
   - Add curriculum-specific logging

### Phase 3: Benchmarking Setup

1. **Evaluation Metrics**
   - **Primary Metrics**:
     - Mean IoU (for correct bbox predictions)
     - Task completion rate (% with valid bbox output)
     - Tool usage rate (when appropriate)
     - Format accuracy (correct <click> syntax)
   
   - **Secondary Metrics**:
     - Response length distribution
     - Entropy over training steps
     - Reward distribution
     - Error types (format vs. accuracy)

2. **Test Sets**
   - **Balanced Test Set**: 50% small, 50% large objects
   - **Real Distribution Test Set**: Original distribution (98.3% small)
   - **Edge Cases Test Set**: Very small (<1%) and very large (>20%) objects

3. **Benchmarking Script**
   ```bash
   # Create benchmark_models.py to evaluate all three models
   python benchmark_models.py \
     --baseline-model "Qwen/Qwen2.5-VL-3B-Instruct" \
     --grpo-model "/path/to/standard_grpo_step_50" \
     --curriculum-model "/path/to/curriculum_grpo_step_50" \
     --test-data "/path/to/test_sets" \
     --output-dir "benchmark_results"
   ```

## Expected Outcomes

### Hypothesis 1: Curriculum Learning Prevents Tool-Only Convergence
- **Expected**: Curriculum model maintains 30-50% bbox attempts
- **Standard GRPO**: <5% bbox attempts (based on current observations)

### Hypothesis 2: Improved Learning Efficiency
- **Expected**: Curriculum model reaches higher IoU faster
- **Metric**: Steps to achieve 0.3 mean IoU

### Hypothesis 3: Better Format Learning
- **Expected**: Curriculum model learns correct <click> format
- **Standard GRPO**: Uses incorrect <boxed> format

## Timeline and Execution Steps

### Week 1: Complete Standard GRPO Training
1. Day 1-2: Resume training from step 25 to 50
2. Day 3: Extract best checkpoint
3. Day 4-5: Initial benchmarking of baseline vs. standard GRPO

### Week 2: Curriculum Learning Implementation
1. Day 1-2: Implement curriculum dataset loader
2. Day 3: Modify reward function for curriculum
3. Day 4-5: Start curriculum training (steps 0-25)
4. Day 6-7: Complete curriculum training (steps 26-50)

### Week 3: Benchmarking and Analysis
1. Day 1-2: Run comprehensive benchmarks
2. Day 3-4: Statistical analysis of results
3. Day 5-7: Write research report with findings

## Critical Implementation Notes

### Memory Safety
```bash
# Use safe configuration for all training
export CUDA_LAUNCH_BLOCKING=1
data.train_batch_size=8
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
```

### Checkpoint Management
```bash
# Save checkpoints every 5 steps for detailed analysis
trainer.save_freq=5
trainer.max_actor_ckpt_to_keep=10
```

### Logging Requirements
- Log all curriculum stage transitions
- Track per-stage metrics separately
- Save example outputs at each stage

## Success Criteria

1. **Curriculum model achieves >30% bbox output rate** (vs. <5% for standard)
2. **Mean IoU improvement of >0.2** over standard GRPO
3. **Correct format usage in >80% of bbox attempts**
4. **Stable entropy throughout training** (>0.3)

## Research Output

### Final Deliverables
1. **Technical Report** including:
   - Comparative analysis of all three models
   - Learning curves and convergence patterns
   - Statistical significance tests
   - Qualitative analysis of output examples

2. **Model Artifacts**:
   - Best checkpoints for both trained models
   - Training logs and metrics
   - Benchmark results dataset

3. **Reproducibility Package**:
   - All training scripts
   - Dataset preparation code
   - Evaluation scripts
   - Configuration files

## Next Immediate Steps

1. **Download current checkpoint** (step 25):
   ```bash
   rsync -avz --progress -e "ssh -p 20014 -i /Users/arc-aman/Desktop/Key" \
     user@147.185.40.11:/home/user/verl/outputs/arc_vision/global_step_25/ \
     ~/arc_vision_checkpoints/standard_grpo_step_25/
   ```

2. **Prepare curriculum dataset balancer**:
   - Implement `prepare_curriculum_stages.py`
   - Create stage-specific parquet files

3. **Set up benchmark infrastructure**:
   - Create evaluation scripts
   - Prepare test datasets
   - Set up metrics tracking

This research plan provides a systematic approach to evaluating curriculum learning's impact on preventing reward exploitation in multi-turn RL training for vision-language models.