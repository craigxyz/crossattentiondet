# Hyperparameter Configurations

**Document Version:** 1.0
**Last Updated:** 2025-11-07

---

## Overview

This document provides a comprehensive reference of all hyperparameter configurations used across different training scripts in the CrossAttentionDet project. It includes baseline training, CSSA ablations, and GAFF ablations.

---

## Table of Contents

1. [Baseline Backbone Training](#1-baseline-backbone-training)
2. [CSSA Ablation Training](#2-cssa-ablation-training)
3. [GAFF Ablation Training](#3-gaff-ablation-training)
4. [Data Augmentation](#4-data-augmentation)
5. [Optimizer Configurations](#5-optimizer-configurations)
6. [Learning Rate Schedules](#6-learning-rate-schedules)
7. [Loss Functions](#7-loss-functions)
8. [Memory Optimization](#8-memory-optimization)

---

## 1. Baseline Backbone Training

### 1.1 Original Configuration (mit_b0, mit_b1)

**Script:** `scripts/train_all_backbones.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Training** | | |
| Epochs | 15 | Fixed for baseline |
| Batch size | 16 | Original config |
| Learning rate | 0.02 | High initial LR |
| Gradient accumulation | 1 | No accumulation |
| Mixed precision | False | FP32 training |
| Gradient checkpointing | False | Not used |
| **Optimizer** | | |
| Type | SGD | Standard stochastic gradient descent |
| Momentum | 0.9 | Standard value |
| Weight decay | 5e-4 | L2 regularization |
| Nesterov | False | Standard momentum |
| **LR Schedule** | | |
| Type | StepLR | Step decay |
| Step size | 15 epochs | Decay at end |
| Gamma | 0.1 | 10× reduction |
| Warmup | None | No warmup |
| **Data** | | |
| Train/test split | 0.8/0.2 | 80% training |
| Num workers | 4 | Data loading threads |
| Pin memory | True | CUDA optimization |
| Shuffle | True | Random batch order |

### 1.2 Memory-Optimized Configuration (mit_b2, mit_b4, mit_b5)

**Script:** `scripts/train_all_backbones.py` (updated)

| Parameter | mit_b2 | mit_b4 | mit_b5 | Notes |
|-----------|--------|--------|--------|-------|
| **Training** | | | | |
| Epochs | 15 | 15 | 15 | Same as original |
| Batch size | 4 | 2 | 2 | Reduced for memory |
| Gradient accumulation | 4 | 8 | 8 | Effective batch = 16 |
| Learning rate | 0.02 | 0.02 | 0.02 | Unchanged |
| Mixed precision | False | True (planned) | True (planned) | FP16 for b4/b5 |
| Gradient checkpointing | False | True (planned) | True (planned) | If still OOM |
| **Memory** | | | | |
| Estimated peak (GB) | ~35 | ~55 | ~65 | On A100 40GB |
| Status | Training | Failed (OOM) | Failed (OOM) | As of 2025-11-07 |

**Effective Training Configuration:**
```python
# Actual batch size per GPU step
physical_batch_size = 4  # or 2 for larger models

# Gradient accumulation to maintain effective batch size
gradient_accumulation_steps = 16 // physical_batch_size

# Effective batch size matches original
effective_batch_size = physical_batch_size * gradient_accumulation_steps  # = 16
```

---

## 2. CSSA Ablation Training

### 2.1 Standard Configuration

**Script:** `crossattentiondet/ablations/scripts/train_cssa_ablation.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Training** | | |
| Epochs | 25 | Default, configurable |
| Batch size | 2 | Low due to multi-modal data |
| Learning rate | 0.005 | Lower than baseline |
| Gradient accumulation | 1 | Can be increased if needed |
| Mixed precision | False | FP32 |
| **Optimizer** | | |
| Type | SGD | Same as baseline |
| Momentum | 0.9 | Standard |
| Weight decay | 1e-4 | 0.0001, lighter regularization |
| **LR Schedule** | | |
| Type | None | Constant learning rate |
| Warmup | None | No warmup |
| Step decay | None | No decay (constant LR) |
| **CSSA-Specific** | | |
| Switching threshold | 0.5 (Phase 1) | Balanced switching |
| | 0.3, 0.7 (Phase 2) | Aggressive/conservative |
| Kernel size | 3 | ECA 1D conv kernel |
| Fusion stages | Variable | [1], [2], [3], [4], [2,3], [3,4], [1,2,3,4] |

### 2.2 Ablation-Specific Parameters

**Phase 1: Stage Selection**

| Experiment | Stages | Threshold | Kernel | Expected Time (hrs) |
|------------|--------|-----------|--------|---------------------|
| exp_001 | [1] | 0.5 | 3 | 3.5 |
| exp_002 | [2] | 0.5 | 3 | 3.5 |
| exp_003 | [3] | 0.5 | 3 | 3.5 |
| exp_004 | [4] | 0.5 | 3 | 3.5 |
| exp_005 | [2,3] | 0.5 | 3 | 3.6 |
| exp_006 | [3,4] | 0.5 | 3 | 3.6 |
| exp_007 | [1,2,3,4] | 0.5 | 3 | 3.7 |

**Phase 2: Threshold Tuning** (TBD based on Phase 1 results)

| Experiment | Stages | Threshold | Expected Behavior |
|------------|--------|-----------|-------------------|
| exp_008 | Top config #1 | 0.3 | More aggressive channel switching |
| exp_009 | Top config #1 | 0.7 | More conservative switching |
| exp_010 | Top config #2 | 0.3 | More aggressive switching |
| exp_011 | Top config #2 | 0.7 | More conservative switching |

---

## 3. GAFF Ablation Training

### 3.1 Standard Configuration

**Script:** `crossattentiondet/ablations/scripts/train_gaff_ablation.py`

| Parameter | Planned | Actual Runs | Notes |
|-----------|---------|-------------|-------|
| **Training** | | | |
| Epochs | 25 | 15 | Actual runs used 15 for speed |
| Batch size | 8 | 16 | Actual runs used 16 |
| Learning rate | 0.0001 | 0.02 | Actual runs used higher LR |
| Gradient accumulation | 1 | 1 | No accumulation |
| Mixed precision | False | False | FP32 |
| **Optimizer** | | | |
| Type | SGD | SGD | Consistent |
| Momentum | 0.9 | 0.9 | Standard |
| Weight decay | 1e-4 | 1e-4 | 0.0001 |
| **LR Schedule** | | | |
| Type | None (planned) | StepLR | Step decay in actual runs |
| Step size | - | 15 epochs | Decay at end |
| Gamma | - | 0.1 | 10× reduction |
| **GAFF-Specific** | | | |
| SE reduction | 4 (Phase 1) | 4 | Bottleneck ratio |
| | 4, 8 (Phase 2) | - | Hyperparameter sweep |
| Inter-modality shared | False (Phase 1) | False | Separate convs |
| | False, True (Phase 2) | - | Hyperparameter sweep |
| Merge bottleneck | False (Phase 1) | False | Direct merge |
| | False, True (Phase 2) | - | Hyperparameter sweep |
| Fusion stages | Variable | Variable | Same as CSSA |

### 3.2 Ablation-Specific Parameters

**Phase 1: Stage Selection**

| Experiment | Stages | SE_r | Inter_Shared | Merge_BN | Time (hrs) |
|------------|--------|------|--------------|----------|------------|
| exp_001 | [1] | 4 | False | False | 3.4 |
| exp_002 | [2] | 4 | False | False | 3.4 |
| exp_003 | [3] | 4 | False | False | 3.4 |
| exp_004 | [4] | 4 | False | False | 3.4 |
| exp_005 | [2,3] | 4 | False | False | 3.6 |
| exp_006 | [3,4] | 4 | False | False | 3.6 |
| exp_007 | [2,3,4] | 4 | False | False | 3.7 |
| exp_008 | [1,2,3,4] | 4 | False | False | 3.8 |

**Phase 2: Hyperparameter Grid** (for each top-3 stage config)

| SE_reduction | Inter_Shared | Merge_BN | Params Impact | Speed Impact |
|--------------|--------------|----------|---------------|--------------|
| 4 | False | False | Baseline | Baseline |
| 4 | False | True | +C² per stage | +5% |
| 4 | True | False | Same as baseline | ~Same |
| 4 | True | True | +C² per stage | +5% |
| 8 | False | False | -C²/2 per stage | -3% |
| 8 | False | True | +C²/2 per stage | +2% |
| 8 | True | False | -C²/2 per stage | -3% |
| 8 | True | True | +C²/2 per stage | +2% |

---

## 4. Data Augmentation

### 4.1 Training Augmentations

**Current Implementation:** Minimal augmentation

| Augmentation | Applied | Parameters | Notes |
|--------------|---------|------------|-------|
| Random horizontal flip | No | - | Could improve generalization |
| Random crop | No | - | May help multi-scale detection |
| Color jitter | No | - | Relevant for RGB modality |
| Random rotation | No | - | May hurt bounding box accuracy |
| Normalization | Yes | See below | Per-modality statistics |

### 4.2 Normalization Parameters

**RGB Modality:**
```python
mean = [0.485, 0.456, 0.406]  # ImageNet statistics
std = [0.229, 0.224, 0.225]
```

**Thermal Modality:**
```python
mean = [0.5]  # Single channel
std = [0.5]
# Or dataset-specific statistics (to be computed)
```

**Event Modality:**
```python
mean = [0.5]  # Single channel
std = [0.5]
# Or dataset-specific statistics (to be computed)
```

**Auxiliary (Thermal + Event stacked):**
```python
# Concatenated along channel dimension
mean = [0.5, 0.5]  # 2 channels
std = [0.5, 0.5]
```

### 4.3 Test-Time Transformations

**No augmentation at test time:**
- Only normalization applied
- No TTA (test-time augmentation) used
- Consistent with training preprocessing

---

## 5. Optimizer Configurations

### 5.1 SGD Configuration (All Experiments)

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,  # Variable by experiment
    momentum=0.9,
    weight_decay=weight_decay,  # Variable by experiment
    nesterov=False
)
```

**Comparison Across Experiments:**

| Experiment Type | Learning Rate | Weight Decay | Momentum |
|-----------------|---------------|--------------|----------|
| Baseline (original) | 0.02 | 5e-4 | 0.9 |
| CSSA Ablations | 0.005 | 1e-4 | 0.9 |
| GAFF Ablations (planned) | 0.0001 | 1e-4 | 0.9 |
| GAFF Ablations (actual) | 0.02 | 1e-4 | 0.9 |

**Design Rationale:**
- **Baseline:** High LR (0.02) for fast convergence in 15 epochs
- **CSSA:** Lower LR (0.005) for stable fusion learning over 25 epochs
- **GAFF (planned):** Very low LR (0.0001) due to complex attention mechanisms
- **GAFF (actual):** High LR (0.02) to match baseline, compensated by scheduler

### 5.2 Alternative Optimizers (Not Currently Used)

**AdamW Configuration (future consideration):**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

**Benefits:**
- Adaptive learning rates per parameter
- Better for attention-based models (GAFF)
- Less sensitive to LR tuning

**Drawbacks:**
- Higher memory usage (stores first/second moments)
- May require different hyperparameter tuning

---

## 6. Learning Rate Schedules

### 6.1 StepLR (Baseline, GAFF Actual Runs)

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=15,  # Decay after 15 epochs
    gamma=0.1      # Multiply LR by 0.1
)
```

**Schedule Visualization:**
```
Epoch:  1-14:  LR = 0.02
Epoch: 15:     LR = 0.002 (step decay)
```

**Use Case:** Short training runs (15 epochs), simple single-step decay

### 6.2 Constant LR (CSSA, GAFF Planned)

```python
# No scheduler
scheduler = None
```

**Schedule Visualization:**
```
Epoch:  1-25:  LR = 0.005 (CSSA) or 0.0001 (GAFF)
```

**Use Case:** Longer training (25 epochs), stable learning without decay

### 6.3 Alternative Schedules (Not Currently Used)

**CosineAnnealingLR:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)
```

**Polynomial Decay:**
```python
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer,
    total_iters=epochs,
    power=0.9
)
```

**Warmup + Cosine:**
```python
# Custom implementation
warmup_epochs = 5
for epoch in range(1, epochs+1):
    if epoch <= warmup_epochs:
        lr = base_lr * (epoch / warmup_epochs)
    else:
        lr = base_lr * 0.5 * (1 + cos(π * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
```

---

## 7. Loss Functions

### 7.1 Faster R-CNN Multi-Task Loss

**Total Loss:**
```python
total_loss = λ_cls * loss_classifier + λ_box * loss_box_reg + λ_obj * loss_objectness + λ_rpn * loss_rpn_box_reg
```

**Default Weights (λ):**
```python
λ_cls = 1.0       # Classification loss
λ_box = 1.0       # Box regression loss
λ_obj = 1.0       # RPN objectness loss
λ_rpn = 1.0       # RPN box regression loss
```

### 7.2 Component Loss Functions

**Classification Loss (Cross-Entropy):**
```python
loss_classifier = F.cross_entropy(
    class_logits,  # (N, num_classes)
    labels,        # (N,)
    reduction='mean'
)
```

**Box Regression Loss (Smooth L1):**
```python
loss_box_reg = F.smooth_l1_loss(
    box_predictions,  # (N, 4)
    box_targets,      # (N, 4)
    beta=1.0,
    reduction='mean'
)
```

**Objectness Loss (Binary Cross-Entropy):**
```python
loss_objectness = F.binary_cross_entropy_with_logits(
    objectness_logits,  # (N,)
    objectness_targets, # (N,)
    reduction='mean'
)
```

**RPN Box Regression Loss (Smooth L1):**
```python
loss_rpn_box_reg = F.smooth_l1_loss(
    rpn_box_predictions,  # (N, 4)
    rpn_box_targets,      # (N, 4)
    beta=1.0,
    reduction='mean'
)
```

### 7.3 Loss Weighting Across Experiments

| Experiment Type | λ_cls | λ_box | λ_obj | λ_rpn | Notes |
|-----------------|-------|-------|-------|-------|-------|
| Baseline | 1.0 | 1.0 | 1.0 | 1.0 | Default Faster R-CNN |
| CSSA | 1.0 | 1.0 | 1.0 | 1.0 | Same as baseline |
| GAFF | 1.0 | 1.0 | 1.0 | 1.0 | Same as baseline |

**No loss reweighting currently applied.** All experiments use standard Faster R-CNN loss.

---

## 8. Memory Optimization

### 8.1 Gradient Accumulation

**Implementation:**
```python
accumulation_steps = 4  # or 8 for larger models

for batch_idx, (images, targets) in enumerate(train_loader):
    # Forward pass
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    # Scale loss by accumulation steps
    losses = losses / accumulation_steps

    # Backward pass
    losses.backward()

    # Update weights only every N steps
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Configuration by Model:**

| Model | Batch Size | Accumulation Steps | Effective Batch Size |
|-------|------------|--------------------|----------------------|
| mit_b0 | 16 | 1 | 16 |
| mit_b1 | 16 | 1 | 16 |
| mit_b2 | 4 | 4 | 16 |
| mit_b4 | 2 | 8 | 16 |
| mit_b5 | 2 | 8 | 16 |

### 8.2 Mixed Precision Training (Planned for mit_b4, mit_b5)

**PyTorch AMP Implementation:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_idx, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

    # Backward pass with scaled gradients
    scaler.scale(losses).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Expected Memory Savings:**
- FP16 activations: ~40-50% memory reduction
- Gradient storage: ~40-50% memory reduction
- Parameters: Still stored in FP32 (master copy)

**Overall:** ~30-40% total memory reduction

### 8.3 Gradient Checkpointing (Planned if Needed)

**PyTorch Implementation:**
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBackbone(nn.Module):
    def forward(self, x):
        # Checkpoint expensive operations
        x = checkpoint(self.stage1, x)
        x = checkpoint(self.stage2, x)
        x = checkpoint(self.stage3, x)
        x = checkpoint(self.stage4, x)
        return x
```

**Trade-off:**
- **Memory:** ~50% reduction (recompute forward pass during backward)
- **Speed:** ~20-30% slower (due to recomputation)

**Use Case:** Last resort for mit_b4, mit_b5 if mixed precision + gradient accumulation insufficient

### 8.4 Memory Usage Summary

**Estimated Peak Memory (A100 40GB):**

| Configuration | Model Params | Activations | Gradients | Optimizer | Total | Fits? |
|---------------|--------------|-------------|-----------|-----------|-------|-------|
| **mit_b0 (BS=16, FP32)** | 1.4 GB | 18 GB | 1.4 GB | 1.4 GB | 22.2 GB | ✅ Yes |
| **mit_b1 (BS=16, FP32)** | 1.7 GB | 22 GB | 1.7 GB | 1.7 GB | 27.1 GB | ✅ Yes |
| **mit_b2 (BS=16, FP32)** | 2.0 GB | 35 GB | 2.0 GB | 2.0 GB | 41.0 GB | ❌ No |
| **mit_b2 (BS=4, FP32)** | 2.0 GB | 9 GB | 2.0 GB | 2.0 GB | 15.0 GB | ✅ Yes |
| **mit_b4 (BS=16, FP32)** | 3.8 GB | 60 GB | 3.8 GB | 3.8 GB | 71.4 GB | ❌ No |
| **mit_b4 (BS=2, FP16)** | 3.8 GB | 9 GB | 2.3 GB | 3.8 GB | 18.9 GB | ✅ Yes |
| **mit_b5 (BS=2, FP16)** | 4.8 GB | 11 GB | 2.9 GB | 4.8 GB | 23.5 GB | ✅ Yes |

**Note:** These are rough estimates. Actual memory usage depends on image resolution, number of objects per image, and framework overhead.

---

## 9. Hyperparameter Tuning Guidelines

### 9.1 Learning Rate Selection

**Rule of Thumb:**
```
Initial LR ∝ sqrt(batch_size)
```

**Examples:**
- Batch size 16: LR = 0.02
- Batch size 8: LR = 0.014
- Batch size 4: LR = 0.01
- Batch size 2: LR = 0.007

**For Gradient Accumulation:**
- Use LR based on *effective* batch size, not physical batch size
- Example: BS=4, accumulation=4 → effective BS=16 → LR=0.02

### 9.2 Weight Decay Selection

**Scaling with Model Size:**
```
Weight decay ∝ 1 / num_parameters
```

**Recommendations:**
- Small models (<100M params): 5e-4 to 1e-3
- Medium models (100-200M params): 1e-4 to 5e-4
- Large models (>200M params): 1e-5 to 1e-4

### 9.3 Batch Size Selection

**Considerations:**
1. **Memory:** Largest batch size that fits in GPU memory
2. **Gradient noise:** Smaller batches → noisier gradients → may help generalization
3. **Training speed:** Larger batches → fewer steps per epoch

**Recommendations:**
- Start with largest batch size that fits
- If overfitting: Reduce batch size
- If underfitting: Increase batch size (use gradient accumulation if needed)

### 9.4 Epoch Count Selection

**Convergence Analysis:**
```python
# Monitor validation loss
if val_loss hasn't improved for 5 epochs:
    Consider early stopping
```

**Recommendations:**
- Baseline (no fusion): 15-20 epochs sufficient
- CSSA fusion: 25-30 epochs (fusion adds complexity)
- GAFF fusion: 30-40 epochs (richer attention requires more training)

---

## 10. Configuration File Templates

### 10.1 CSSA Ablation Config (JSON)

```json
{
  "experiment_id": "exp_003_s3_t0.5",
  "backbone": "mit_b1",
  "fusion_type": "cssa",
  "training": {
    "epochs": 25,
    "batch_size": 2,
    "learning_rate": 0.005,
    "optimizer": {
      "type": "SGD",
      "momentum": 0.9,
      "weight_decay": 0.0001
    },
    "lr_scheduler": null
  },
  "cssa": {
    "stages": [3],
    "switching_thresh": 0.5,
    "kernel_size": 3
  },
  "data": {
    "data_dir": "../RGBX_Semantic_Segmentation/data/images",
    "labels_dir": "../RGBX_Semantic_Segmentation/data/labels",
    "train_split": 0.8
  }
}
```

### 10.2 GAFF Ablation Config (JSON)

```json
{
  "experiment_id": "exp_003_s3_r4_is0_mb0",
  "backbone": "mit_b1",
  "fusion_type": "gaff",
  "training": {
    "epochs": 25,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "optimizer": {
      "type": "SGD",
      "momentum": 0.9,
      "weight_decay": 0.0001
    },
    "lr_scheduler": null
  },
  "gaff": {
    "stages": [3],
    "se_reduction": 4,
    "inter_shared": false,
    "merge_bottleneck": false
  },
  "data": {
    "data_dir": "../RGBX_Semantic_Segmentation/data/images",
    "labels_dir": "../RGBX_Semantic_Segmentation/data/labels",
    "train_split": 0.8
  }
}
```

### 10.3 Baseline Config (JSON)

```json
{
  "experiment_id": "baseline_mit_b1",
  "backbone": "mit_b1",
  "fusion_type": null,
  "training": {
    "epochs": 15,
    "batch_size": 16,
    "learning_rate": 0.02,
    "optimizer": {
      "type": "SGD",
      "momentum": 0.9,
      "weight_decay": 0.0005
    },
    "lr_scheduler": {
      "type": "StepLR",
      "step_size": 15,
      "gamma": 0.1
    }
  },
  "data": {
    "data_dir": "../RGBX_Semantic_Segmentation/data/images",
    "labels_dir": "../RGBX_Semantic_Segmentation/data/labels",
    "train_split": 0.8
  }
}
```

---

## 11. Quick Reference Table

### All Configurations at a Glance

| Experiment | Epochs | Batch | LR | Weight Decay | LR Schedule | Fusion Params |
|------------|--------|-------|----|--------------| ------------|---------------|
| **Baseline (b0,b1)** | 15 | 16 | 0.02 | 5e-4 | StepLR(15,0.1) | N/A |
| **Baseline (b2)** | 15 | 4+4acc | 0.02 | 5e-4 | StepLR(15,0.1) | N/A |
| **CSSA (all)** | 25 | 2 | 0.005 | 1e-4 | None | threshold ∈ {0.3,0.5,0.7} |
| **GAFF (planned)** | 25 | 8 | 0.0001 | 1e-4 | None | SE_r∈{4,8}, inter_s∈{F,T}, merge_bn∈{F,T} |
| **GAFF (actual)** | 15 | 16 | 0.02 | 1e-4 | StepLR(15,0.1) | SE_r=4, inter_s=F, merge_bn=F |

---

## 12. Best Practices & Recommendations

### 12.1 General Guidelines

1. **Start with defaults:**
   - Use baseline config as starting point
   - Modify only when necessary

2. **One variable at a time:**
   - Change LR → observe → change batch size → observe
   - Don't change multiple hyperparameters simultaneously

3. **Monitor validation metrics:**
   - Training loss alone is insufficient
   - Watch for overfitting (train/val gap)

4. **Use checkpointing:**
   - Save best model by validation mAP
   - Save periodic checkpoints (every 5 epochs)

### 12.2 Debugging Failed Experiments

**High training loss (>1.0):**
- Reduce learning rate by 10×
- Check data normalization
- Verify loss function weights

**NaN loss:**
- Reduce learning rate dramatically (0.001 or lower)
- Enable gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)`
- Check for division by zero in custom loss functions

**OOM errors:**
1. Reduce batch size
2. Enable gradient accumulation
3. Enable mixed precision (FP16)
4. Enable gradient checkpointing
5. Reduce image resolution (last resort)

**Slow training:**
- Increase num_workers for data loading
- Enable pin_memory
- Use larger batch size with gradient accumulation
- Check GPU utilization with `nvidia-smi`

---

## Appendix: Script Command Examples

### Baseline Training
```bash
python scripts/train_all_backbones.py \
    --backbone mit_b1 \
    --epochs 15 \
    --batch-size 16 \
    --lr 0.02
```

### CSSA Ablation
```bash
python crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --backbone mit_b1 \
    --epochs 25 \
    --batch-size 2 \
    --lr 0.005 \
    --cssa-stages "3" \
    --cssa-thresh 0.5 \
    --output-dir results/cssa_ablations/exp_003
```

### GAFF Ablation
```bash
python crossattentiondet/ablations/scripts/train_gaff_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --backbone mit_b1 \
    --epochs 25 \
    --batch-size 8 \
    --lr 0.0001 \
    --gaff-stages "3" \
    --gaff-se-reduction 4 \
    --gaff-inter-shared false \
    --gaff-merge-bottleneck false \
    --output-dir results/gaff_ablations_full/exp_003
```

---

**Document End**

**For implementation details, refer to:**
- Training scripts in `crossattentiondet/ablations/scripts/`
- Config module: `crossattentiondet/config.py`
- Model definitions: `crossattentiondet/models/`
