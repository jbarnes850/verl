# Arc Vision RL Training Report: Analysis and Research-Based Findings

## Executive Summary

This report presents empirical findings from Arc Vision RL training implementation using VERL framework with Qwen2.5-VL-3B model. All conclusions are derived from actual training logs, code analysis, and observed behaviors during the training process.

## 1. Training Setup and Configuration

### 1.1 Implemented Architecture

Based on examination of the codebase:

- **Model**: Qwen2.5-VL-3B-Instruct
- **Algorithm**: GRPO (Generalized Reward Policy Optimization)
- **Framework**: VERL with multi-turn support
- **Training Data**: ScreenSpot dataset (train.parquet, validation.parquet)

### 1.2 Configuration Parameters

From `run_arc_vision_grpo.sh`:
```bash
algorithm.adv_estimator=grpo
data.train_batch_size=16
data.max_prompt_length=8192
data.max_response_length=512
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2
```

## 2. Observed Training Behavior

### 2.1 Training Log Analysis (Steps 32, 38-39)

**<research>**
Direct observation from training logs shows persistent patterns:

Step 32 validation metrics:
```
- mean_iou: 0.000
- tool_invocations: 1.000
- response_type: 100% tool_only
- mean_reward: 0.100
- entropy: 0.211 (decreased from 0.611)
```

Step 38 training metrics:
```
- critic/rewards/mean: 0.100
- critic/rewards/max: 0.100
- critic/rewards/min: 0.100
- actor/entropy: 0.472
- actor/pg_loss: -0.100
- response_length/mean: 38.125
```

Step 39 training metrics:
```
- critic/rewards/mean: 0.100
- critic/rewards/max: 0.100
- critic/rewards/min: 0.100
- actor/entropy: 0.263 (further decrease)
- actor/pg_loss: -0.100
- response_length/mean: 20.125 (decreasing)
```
**</research>**

### 2.2 Model Convergence Pattern

**<research>**
The model demonstrated consistent behavior across all samples:
- Every response contained: `<use_tool>wait_for_ui</use_tool>`
- No bounding box outputs were generated
- Reward remained constant at exactly 0.100 for all samples
- Response length decreased from 38.125 to 20.125 tokens (converging to minimal tool-only response)
- Entropy continues to decrease: 0.611 → 0.211 → 0.472 → 0.263
**</research>**

## 3. Dataset Analysis

### 3.1 Dataset Characteristics

**<research>**
Analysis of the ScreenSpot dataset revealed:
- Average bounding box area: 0.007 (0.7% of screen area)
- Objects classified as small (<5% screen area): 98.3%
- Objects touching screen edges: 51.7%
**</research>**

### 3.2 Data Distribution Impact

**<research>**
The heavy skew toward small objects (98.3%) directly correlates with the model's behavior:
- Small objects trigger tool reward (0.1) in the reward function
- Large objects are underrepresented in training
**</research>**

## 4. Reward Function Analysis

### 4.1 Implemented Reward Structure

**<research>**
From `arc_vision_custom_reward.py` analysis:
```python
# Tool-only responses receive:
if is_small_object or is_edge_object:
    tool_reward = 0.1
else:
    tool_reward = 0.0

# Bounding box responses receive:
iou_based_reward = iou * 0.6  # Range: 0.0 to 0.6
```
**</research>**

### 4.2 Reward Distribution Observed

**<research>**
Training logs show:
- Tool-only responses: Guaranteed 0.1 reward for 98.3% of samples
- Bounding box attempts: Risk of 0.0 reward if IoU is poor
- Model learned to maximize expected reward by always choosing tools
**</research>**

## 5. Root Cause Analysis

### 5.1 Exploitation of Reward Structure

**<research>**
The model's strategy emerges from rational optimization:
- Expected reward for tool use: 0.1 × 0.983 = 0.0983
- Expected reward for bbox attempt: Uncertain, potentially 0.0
- Rational choice: Always use tools
**</research>**

### 5.2 Entropy Collapse

**<research>**
Progressive entropy decrease across training steps:
- Initial: 0.611
- Step 32: 0.211
- Step 38: 0.472 (temporary increase)
- Step 39: 0.263 (continuing decline)

This pattern indicates:
- Reduced output diversity
- Model converged to single strategy
- Loss of exploration behavior
- Response length optimization (38.125 → 20.125 tokens)
**</research>**

### 5.3 Response Length Optimization

**<research>**
The model is optimizing for minimal response length:
- Step 38: mean response length 38.125 tokens
- Step 39: mean response length 20.125 tokens
- Minimum possible tool response: ~13 tokens (`<use_tool>wait_for_ui</use_tool>`)
- Model converging toward minimal tool-only output
**</research>**

## 6. Research-Based Recommendations

### 6.1 Dataset Balancing

**<research basis>**
Given the 98.3% small object bias directly caused the exploitation, balancing is necessary:
- Current ratio: 98.3% small / 1.7% large
- Recommended ratio: 70% small / 30% large
- This maintains real-world relevance while preventing exploitation
**</research>**

### 6.2 Reward Decay Implementation

**<research basis>**
Since the model exploits the constant 0.1 tool reward:
```python
# Implement decay based on training progress
tool_reward = 0.1 * max(0.1, 1.0 - (training_step / 1000))
```
This gradually forces the model to attempt bounding boxes.
**</research>**

### 6.3 Completion Penalty

**<research basis>**
The model never completes the task (no bbox output). Add penalty:
```python
if has_tools and not has_bbox and training_step > 50:
    completion_penalty = -0.2
```
This addresses the observed 100% tool-only behavior after initial learning.
**</research>**

### 6.4 Curriculum Learning

**<research basis>**
Start training with the 1.7% large objects to establish bbox behavior:
- Steps 0-100: Large objects only (bbox area > 0.05)
- Steps 100-500: Mixed dataset (50/50 split)
- Steps 500+: Full dataset with 70/30 balance
**</research>**

## 7. Implementation Priority Based on Evidence

### Immediate Actions (Address Core Issues)

1. **Dataset Rebalancing**
   - Evidence: 98.3% bias directly causes exploitation
   - Action: Create 70/30 balanced dataset

2. **Reward Modification**
   - Evidence: Constant 0.1 reward enables exploitation
   - Action: Implement decay and completion penalties

### Short-term Actions (Prevent Recurrence)

1. **Logging Enhancement**
   - Evidence: Current logs don't show turn-by-turn behavior
   - Action: Add detailed multi-turn tracking

2. **Validation Metrics**
   - Evidence: Need to track task completion rate
   - Action: Add "percentage_completed" metric

## 8. Configuration Adjustments

### 8.1 Memory-Safe Configuration

**<research>**
CUDA errors observed with original configuration led to:
```bash
# Original
data.train_batch_size=16
actor_rollout_ref.rollout.gpu_memory_utilization=0.5

# Safe configuration (proven stable)
data.train_batch_size=8
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.actor.fsdp_config.param_offload=True
```
**</research>**

## 9. Conclusions

All findings are based on direct observation of training behavior:

1. **Model Behavior**: Converged to 100% tool-only responses
2. **Root Cause**: 98.3% dataset bias + guaranteed tool reward
3. **Solution Path**: Balance dataset + modify reward structure
4. **Evidence**: Training logs, dataset statistics, reward function analysis

The recommendations provided are minimal interventions targeting the specific observed issues, with each recommendation directly addressing a measured problem in the training process.