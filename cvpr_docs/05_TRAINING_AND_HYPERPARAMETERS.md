# Training & Hyperparameters

**Complete Training Configuration and Optimization Strategies**

[← Back to Index](00_INDEX.md) | [← Previous: Ablation Studies](04_ABLATION_STUDIES.md) | [Next: Implementation Details →](06_IMPLEMENTATION_DETAILS.md)

---

## Training Configuration

### Baseline Hyperparameters

```python
# From crossattentiondet/config.py
num_epochs = 15  # Baseline, 25 for ablations
batch_size = 2   # Varies: 2-32 depending on backbone
learning_rate = 0.005  # Base LR, scaled for larger batches
momentum = 0.9
weight_decay = 0.0005

# Optimizer
optimizer = SGD(model.parameters(), 
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay)

# LR Schedule (some experiments)
lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
# OR constant LR for short runs
```

### Batch Size Scaling

| Backbone | Batch Size | Gradient Accum | Effective BS | Learning Rate |
|----------|------------|----------------|--------------|---------------|
| mit_b0 | 16 | 1 | 16 | 0.02 |
| mit_b1 | 16 | 1 | 16 | 0.02 |
| mit_b2 | 4 | 4 | 16 | 0.02 |
| mit_b4 | 2 | 8 | 16 | 0.02 |
| mit_b5 | 2 | 8 | 16 | 0.02 |

---

## Loss Functions

### Faster R-CNN Multi-Task Loss

```python
total_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
```

**Components:**
1. **loss_classifier:** Cross-entropy for object classification
2. **loss_box_reg:** Smooth L1 for detection head box regression  
3. **loss_objectness:** BCE for RPN objectness scores
4. **loss_rpn_box_reg:** Smooth L1 for RPN box regression

**Typical Values (from training logs):**
- Total: 0.10-0.13
- Classifier: 0.03-0.05
- Box Reg: 0.02-0.03
- Objectness: 0.02-0.03
- RPN Box Reg: 0.01-0.02

---

## Evaluation Metrics

### COCO Detection Metrics

**Primary:**
- **mAP:** Mean AP @ IoU [0.5:0.95:0.05] (10 thresholds)

**Secondary:**
- **mAP@50:** AP @ IoU 0.5 (lenient)
- **mAP@75:** AP @ IoU 0.75 (strict)

**By Size:**
- **mAP_small:** area < 32²
- **mAP_medium:** 32² < area < 96²
- **mAP_large:** area > 96²

**Recall:**
- **mAR@1, @10, @100:** Max 1/10/100 detections per image

---

## A100 GPU Optimization

**Hardware:** NVIDIA A100 (79.25 GiB memory)

**Memory Optimizations:**
- Gradient accumulation for large backbones
- Mixed precision (FP16) for mit_b4/b5
- Gradient checkpointing (if needed)

**Training Time Estimates (15 epochs):**
- mit_b0: 2.68 hours
- mit_b1: 3.44 hours (best)
- mit_b2: ~6.75 hours (estimated)
- CSSA ablation: ~3 hours
- GAFF ablation: ~3-4 hours

---

For detailed hyperparameter configurations, see `docs/HYPERPARAMETER_CONFIGURATIONS.md`

[← Back to Index](00_INDEX.md) | [← Previous: Ablation Studies](04_ABLATION_STUDIES.md) | [Next: Implementation Details →](06_IMPLEMENTATION_DETAILS.md)
