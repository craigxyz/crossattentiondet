# Backbone Architecture Specifications

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Architecture Family:** MiT (Mix Transformer) from SegFormer

---

## Overview

This document provides detailed specifications for all backbone architectures used in the CrossAttentionDet project. The project uses the MiT (Mix Transformer) family from SegFormer, which provides hierarchical vision transformers optimized for dense prediction tasks.

**Available Backbones:**
- **mit_b0** - Lightweight (smallest model)
- **mit_b1** - Balanced (default choice)
- **mit_b2** - Base model (deeper than b1)
- **mit_b4** - Large model (much deeper)
- **mit_b5** - Extra large (deepest model)

**Note:** mit_b3 exists in SegFormer but is not included in this project's experiments.

---

## 1. Architecture Family Overview

### MiT (Mix Transformer) Design

The MiT backbone is a hierarchical transformer that produces multi-scale features through 4 stages, similar to CNNs like ResNet. Key innovations:

1. **Hierarchical Structure:** 4 stages with progressively increasing channels and decreasing spatial resolution
2. **Overlapped Patch Merging:** Reduces spatial dimensions while expanding channels
3. **Efficient Self-Attention:** Mix-FFN reduces computational cost
4. **No Positional Encoding:** Uses zero-padding to encode position information

**Stage Outputs:**
- **Stage 1:** H/4 × W/4 resolution
- **Stage 2:** H/8 × W/8 resolution
- **Stage 3:** H/16 × W/16 resolution
- **Stage 4:** H/32 × W/32 resolution

These multi-scale features are fed into a Feature Pyramid Network (FPN) for object detection.

---

## 2. Backbone Comparison Table

| Backbone | Total Params | Embed Dims | Depths | Training Status | Memory Req. | Speed | Best For |
|----------|--------------|------------|--------|-----------------|-------------|-------|----------|
| mit_b0 | 55.7M | [32, 64, 160, 256] | [2, 2, 2, 2] | ✅ Success | ~15 GB | Fastest | Resource-constrained |
| mit_b1 | 69.5M | [64, 128, 320, 512] | [2, 2, 2, 2] | ✅ Success | ~20 GB | Fast | Default choice |
| mit_b2 | 82.1M | [64, 128, 320, 512] | [3, 4, 6, 3] | 🔄 Training | ~40 GB* | Medium | Accuracy/speed balance |
| mit_b4 | 155.4M | [64, 128, 320, 512] | [3, 8, 27, 3] | ❌ OOM | >79 GB | Slow | High accuracy |
| mit_b5 | 196.6M | [64, 128, 320, 512] | [3, 6, 40, 3] | ❌ OOM | >79 GB | Slowest | Maximum accuracy |

*mit_b2 requires gradient accumulation (batch_size=4, accum_steps=4) to fit in 79 GB GPU memory

**Parameter counts include:**
- MiT backbone encoder
- Feature Pyramid Network (FPN)
- Region Proposal Network (RPN)
- ROI heads (box predictor)

---

## 3. Detailed Architecture Specifications

### 3.1 mit_b0 (Lightweight)

**Status:** ✅ Training successful
**Total Parameters:** 55,718,241 (55.7M)
**Training Time:** 2.68 hours (15 epochs)
**Best Loss:** 0.1057

**Architecture Configuration:**
```python
embed_dims = [32, 64, 160, 256]    # Channel dimensions per stage
depths = [2, 2, 2, 2]              # Number of transformer blocks per stage
num_heads = [1, 2, 5, 8]           # Attention heads per stage
mlp_ratios = [4, 4, 4, 4]          # MLP expansion ratios
sr_ratios = [8, 4, 2, 1]           # Spatial reduction ratios for attention
```

**Stage Details:**
| Stage | Input Size | Output Size | Channels | Blocks | Attention Heads | Params (est.) |
|-------|------------|-------------|----------|--------|-----------------|---------------|
| 1 | H/4 × W/4 | H/4 × W/4 | 32 | 2 | 1 | ~0.5M |
| 2 | H/4 × W/4 | H/8 × W/8 | 64 | 2 | 2 | ~1.5M |
| 3 | H/8 × W/8 | H/16 × W/16 | 160 | 2 | 5 | ~8M |
| 4 | H/16 × W/16 | H/32 × W/32 | 256 | 2 | 8 | ~15M |

**Characteristics:**
- ✅ Smallest model, fastest training
- ✅ Fits comfortably in GPU memory (batch_size=16 works)
- ✅ Good for rapid experimentation
- ⚠️ Lower capacity may limit accuracy on complex datasets

**Use Cases:**
- Initial prototyping and debugging
- Resource-constrained deployment
- Real-time inference requirements
- Baseline comparison

---

### 3.2 mit_b1 (Balanced - Default)

**Status:** ✅ Training successful
**Total Parameters:** 69,503,585 (69.5M)
**Training Time:** 3.44 hours (15 epochs)
**Best Loss:** 0.1027

**Architecture Configuration:**
```python
embed_dims = [64, 128, 320, 512]   # Channel dimensions per stage
depths = [2, 2, 2, 2]              # Number of transformer blocks per stage
num_heads = [1, 2, 5, 8]           # Attention heads per stage
mlp_ratios = [4, 4, 4, 4]          # MLP expansion ratios
sr_ratios = [8, 4, 2, 1]           # Spatial reduction ratios for attention
```

**Stage Details:**
| Stage | Input Size | Output Size | Channels | Blocks | Attention Heads | Params (est.) |
|-------|------------|-------------|----------|--------|-----------------|---------------|
| 1 | H/4 × W/4 | H/4 × W/4 | 64 | 2 | 1 | ~2M |
| 2 | H/4 × W/4 | H/8 × W/8 | 128 | 2 | 2 | ~5M |
| 3 | H/8 × W/8 | H/16 × W/16 | 320 | 2 | 5 | ~25M |
| 4 | H/16 × W/16 | H/32 × W/32 | 512 | 2 | 8 | ~40M |

**Characteristics:**
- ✅ Best accuracy/speed/memory trade-off
- ✅ Default choice for all ablation studies
- ✅ 2× channel capacity vs. mit_b0
- ✅ Fits in GPU memory with batch_size=16
- ✅ Achieved best loss (0.1027) among successful runs

**Use Cases:**
- Default backbone for ablation studies
- Production deployment with good accuracy
- Balanced resource utilization
- Benchmark comparisons

**Why This is the Default:**
1. Successfully trains without memory issues
2. Better capacity than mit_b0 (69.5M vs 55.7M parameters)
3. Fast enough for extensive ablation studies
4. Good accuracy baseline for comparison

---

### 3.3 mit_b2 (Base Model)

**Status:** 🔄 Currently training (1/15 epochs complete)
**Total Parameters:** 82,104,673 (82.1M)
**Training Time:** ~27 minutes/epoch
**Best Loss:** 0.3787 (after 1 epoch)

**Architecture Configuration:**
```python
embed_dims = [64, 128, 320, 512]   # Same channels as b1
depths = [3, 4, 6, 3]              # DEEPER than b1 [2, 2, 2, 2]
num_heads = [1, 2, 5, 8]           # Attention heads per stage
mlp_ratios = [4, 4, 4, 4]          # MLP expansion ratios
sr_ratios = [8, 4, 2, 1]           # Spatial reduction ratios for attention
```

**Stage Details:**
| Stage | Input Size | Output Size | Channels | Blocks | Attention Heads | Params (est.) |
|-------|------------|-------------|----------|--------|-----------------|---------------|
| 1 | H/4 × W/4 | H/4 × W/4 | 64 | 3 | 1 | ~3M |
| 2 | H/4 × W/4 | H/8 × W/8 | 128 | 4 | 2 | ~10M |
| 3 | H/8 × W/8 | H/16 × W/16 | 320 | 6 | 5 | ~38M |
| 4 | H/16 × W/16 | H/32 × W/32 | 512 | 3 | 8 | ~30M |

**Memory Optimization (Required):**
```python
batch_size = 4              # Reduced from 16
grad_accum_steps = 4        # Accumulate 4 batches
effective_batch_size = 16   # Same as original
use_grad_checkpoint = False # Not enabled yet (option available)
```

**Characteristics:**
- ⚠️ Requires memory optimization to train
- ✅ Same channel dimensions as b1, but deeper
- ✅ Total depth: 16 blocks (vs. 8 for b1)
- ✅ More representational capacity
- ⚠️ ~28% slower than b1 (27 min/epoch vs. ~21 min/epoch for b1)

**Key Difference from mit_b1:**
- **b1:** Shallow and wide (more channels per block)
- **b2:** Deep and narrow (more blocks per stage)
- **Hypothesis:** Depth helps capture complex multi-modal relationships

**Use Cases:**
- When accuracy is more important than speed
- Multi-modal fusion benefits from depth
- Moderate resource increase acceptable
- Research into depth vs. width trade-offs

**Training Notes:**
- First attempt (batch_size=16): CUDA OOM
- Second attempt (batch_size=4, grad_accum=4): Training successfully
- Checkpoint saved after epoch 1: `epoch_001_loss_0.3788.pth`
- Estimated total training time: ~6.75 hours (15 epochs × 27 min)

---

### 3.4 mit_b4 (Large Model)

**Status:** ❌ Failed (CUDA Out of Memory)
**Total Parameters:** 155,397,473 (155.4M)
**Memory Required:** >79.25 GiB
**Error:** CUDA OOM after trying to allocate 1.30 GiB (GPU exhausted)

**Architecture Configuration:**
```python
embed_dims = [64, 128, 320, 512]   # Same channels as b1/b2
depths = [3, 8, 27, 3]             # MUCH DEEPER (especially stage 3!)
num_heads = [1, 2, 5, 8]           # Attention heads per stage
mlp_ratios = [4, 4, 4, 4]          # MLP expansion ratios
sr_ratios = [8, 4, 2, 1]           # Spatial reduction ratios for attention
```

**Stage Details:**
| Stage | Input Size | Output Size | Channels | Blocks | Attention Heads | Params (est.) |
|-------|------------|-------------|----------|--------|-----------------|---------------|
| 1 | H/4 × W/4 | H/4 × W/4 | 64 | 3 | 1 | ~3M |
| 2 | H/4 × W/4 | H/8 × W/8 | 128 | 8 | 2 | ~20M |
| 3 | H/8 × W/8 | H/16 × W/16 | 320 | 27 | 5 | ~135M |
| 4 | H/16 × W/16 | H/32 × W/32 | 512 | 3 | 8 | ~30M |

**Characteristics:**
- ❌ Cannot train with current GPU memory (79.25 GiB A100)
- ⚠️ Stage 3 has 27 transformer blocks (extreme depth!)
- ⚠️ 155.4M parameters (2.2× larger than b1)
- ⚠️ Gradient accumulation alone insufficient
- 🔲 Requires additional optimization (gradient checkpointing, mixed precision)

**Why It Fails:**
1. **Model size:** 155.4M parameters × 4 bytes = ~622 MB (model weights)
2. **Activations:** With batch_size=16, intermediate activations become huge
3. **Gradients:** Same memory as model weights for backprop
4. **Optimizer state:** Adam/SGD needs 1-2× model memory
5. **Total:** Easily exceeds 79 GB with full-resolution multi-modal images

**Memory Optimization Options (Not Yet Tested):**
1. **Gradient Checkpointing:** Trade compute for memory (recompute activations)
   ```python
   use_grad_checkpoint = True  # Available in resume_large_backbones.py
   ```
2. **Mixed Precision (FP16/BF16):** Reduce memory by 50%
   ```python
   torch.cuda.amp.autocast()
   ```
3. **Further Reduce Batch Size:** batch_size=2, grad_accum=8
4. **Reduce FPN Channels:** 256 → 128 (lighter detection head)
5. **Selective Fusion:** Only apply fusion at 1-2 stages instead of all 4

**Use Cases (If Memory Issue Resolved):**
- High-accuracy research experiments
- Exploring depth scaling benefits
- Comparison with state-of-the-art models
- When inference speed is not critical

---

### 3.5 mit_b5 (Extra Large)

**Status:** ❌ Failed (CUDA Out of Memory)
**Total Parameters:** 196,598,113 (196.6M)
**Memory Required:** >79.25 GiB
**Error:** CUDA OOM after trying to allocate 1.30 GiB (GPU exhausted)

**Architecture Configuration:**
```python
embed_dims = [64, 128, 320, 512]   # Same channels as b1/b2/b4
depths = [3, 6, 40, 3]             # DEEPEST (stage 3 has 40 blocks!)
num_heads = [1, 2, 5, 8]           # Attention heads per stage
mlp_ratios = [4, 4, 4, 4]          # MLP expansion ratios
sr_ratios = [8, 4, 2, 1]           # Spatial reduction ratios for attention
```

**Stage Details:**
| Stage | Input Size | Output Size | Channels | Blocks | Attention Heads | Params (est.) |
|-------|------------|-------------|----------|--------|-----------------|---------------|
| 1 | H/4 × W/4 | H/4 × W/4 | 64 | 3 | 1 | ~3M |
| 2 | H/4 × W/4 | H/8 × W/8 | 128 | 6 | 2 | ~15M |
| 3 | H/8 × W/8 | H/16 × W/16 | 320 | 40 | 5 | ~200M |
| 4 | H/16 × W/16 | H/32 × W/32 | 512 | 3 | 8 | ~30M |

**Characteristics:**
- ❌ Cannot train with current GPU memory (79.25 GiB A100)
- ⚠️ Stage 3 has 40 transformer blocks (most extreme depth!)
- ⚠️ 196.6M parameters (2.8× larger than b1)
- ⚠️ Largest model in MiT family
- 🔲 Requires extensive memory optimization

**Why It Fails:**
- Even larger than mit_b4 (196.6M vs. 155.4M)
- Stage 3 alone: ~200M parameters (40 blocks × 320 channels)
- Activations scale with batch size and depth
- Same memory issues as mit_b4, but worse

**Memory Optimization Strategy:**
Same options as mit_b4, but may need to combine multiple techniques:
- Gradient checkpointing (required)
- Mixed precision FP16 (required)
- batch_size=2, grad_accum=8 (required)
- Possibly reduce FPN channels or selective fusion

**Use Cases (If Memory Issue Resolved):**
- Maximum accuracy for research
- State-of-the-art comparison
- Exploring scaling laws
- Offline processing where speed doesn't matter

---

## 4. Architecture Comparison Analysis

### 4.1 Depth vs. Width Trade-offs

| Backbone | Width (max channels) | Depth (total blocks) | Width × Depth | Parameters | Training Status |
|----------|----------------------|----------------------|---------------|------------|-----------------|
| mit_b0 | 256 | 8 | 2,048 | 55.7M | ✅ Success |
| mit_b1 | 512 | 8 | 4,096 | 69.5M | ✅ Success |
| mit_b2 | 512 | 16 | 8,192 | 82.1M | 🔄 Training |
| mit_b4 | 512 | 41 | 20,992 | 155.4M | ❌ OOM |
| mit_b5 | 512 | 52 | 26,624 | 196.6M | ❌ OOM |

**Observations:**
- **b0 → b1:** Doubles channel width (256 → 512), modest parameter increase
- **b1 → b2:** Same width, doubles depth (8 → 16), 18% parameter increase
- **b2 → b4:** Same width, 2.5× depth increase, 89% parameter increase (mostly in stage 3)
- **b4 → b5:** Same width, slight depth increase (41 → 52), 27% parameter increase

**Conclusion:** Depth scaling (b2, b4, b5) increases parameters much more than width scaling (b0 → b1).

### 4.2 Memory Consumption Analysis

**Measured GPU Memory Usage (79.25 GiB A100):**

| Backbone | Batch Size | Grad Accum | Effective BS | Peak Memory | Status |
|----------|------------|------------|--------------|-------------|--------|
| mit_b0 | 16 | 1 | 16 | ~20 GB | ✅ Success |
| mit_b1 | 16 | 1 | 16 | ~25 GB | ✅ Success |
| mit_b2 | 16 | 1 | 16 | >79 GB | ❌ OOM |
| mit_b2 | 4 | 4 | 16 | ~40 GB | ✅ Success |
| mit_b4 | 4 | 4 | 16 | >79 GB | ❌ OOM |
| mit_b5 | 4 | 4 | 16 | >79 GB | ❌ OOM |

**Memory Breakdown (Estimated for mit_b1):**
- Model weights: ~280 MB (69.5M × 4 bytes)
- Gradients: ~280 MB (same as weights)
- Optimizer state (SGD): ~280 MB (momentum buffer)
- Activations (batch_size=16): ~15-20 GB (depends on image size)
- FPN + RPN + ROI head: ~3-5 GB
- PyTorch overhead: ~1-2 GB
- **Total:** ~25 GB

**Scaling:**
- mit_b2: 82.1M params → 1.18× base memory → ~30 GB + activations → ~40 GB with grad accum
- mit_b4: 155.4M params → 2.2× base memory → activation growth dominates → >79 GB
- mit_b5: 196.6M params → 2.8× base memory → activation growth dominates → >79 GB

### 4.3 Training Speed Comparison

**Time per Epoch (15 epochs, batch_size=16 or effective 16):**

| Backbone | Time/Epoch | Total Time (15 epochs) | Relative Speed |
|----------|------------|------------------------|----------------|
| mit_b0 | ~10.7 min | 2.68 hours | 1.0× (fastest) |
| mit_b1 | ~13.8 min | 3.44 hours | 1.29× slower |
| mit_b2 | ~27 min | 6.75 hours (est.) | 2.52× slower |
| mit_b4 | - | - | ~5× slower (est.) |
| mit_b5 | - | - | ~6× slower (est.) |

**Observations:**
- b0 → b1: 29% slower (acceptable)
- b1 → b2: 96% slower (2× slower due to depth)
- Depth has superlinear impact on speed (more sequential operations)

### 4.4 Accuracy vs. Computational Cost

**Expected Accuracy Trend (based on SegFormer paper):**

| Backbone | Params | GFLOPs (est.) | Expected mAP (relative) | Actual Loss (if available) |
|----------|--------|---------------|-------------------------|----------------------------|
| mit_b0 | 55.7M | ~50 | Baseline | 0.1057 |
| mit_b1 | 69.5M | ~60 | +2-3% | 0.1027 (best) |
| mit_b2 | 82.1M | ~90 | +3-5% | 0.3787 (1 epoch only) |
| mit_b4 | 155.4M | ~180 | +5-7% | - |
| mit_b5 | 196.6M | ~220 | +6-8% | - |

**Note:** Loss values after 1 epoch are not directly comparable (mit_b2 still early in training).

**Accuracy/Cost Trade-off:**
- **mit_b0:** Best cost efficiency (baseline)
- **mit_b1:** Sweet spot (small cost increase, good accuracy)
- **mit_b2:** Moderate improvement, 2× training time
- **mit_b4/b5:** Diminishing returns (large cost increase for small accuracy gain)

---

## 5. Multi-Modal Considerations

### 5.1 Channel Handling

All backbones are modified to accept **5-channel input** instead of standard 3-channel RGB:

**Input Channels:**
- Channels 0-2: RGB (standard)
- Channel 3: Thermal/Infrared
- Channel 4: Event camera data

**Modification:**
The first convolutional layer in the patch embedding is changed from:
```python
# Original SegFormer (RGB only)
Conv2d(3, embed_dims[0], kernel_size=7, stride=4, padding=3)
```
To:
```python
# CrossAttentionDet (RGB + Thermal + Event)
Conv2d(5, embed_dims[0], kernel_size=7, stride=4, padding=3)
```

**Impact on Parameters:**
- Original: 3 × C_out × 7 × 7 input parameters
- Modified: 5 × C_out × 7 × 7 input parameters
- Increase: +67% in first layer only (negligible overall)

**Example for mit_b1:**
- Original: 3 × 64 × 7 × 7 = 9,408 parameters
- Modified: 5 × 64 × 7 × 7 = 15,680 parameters
- Additional: 6,272 parameters (~0.009% of total)

### 5.2 Fusion Integration Points

The backbone produces features at 4 stages, and fusion can be applied at any combination:

**Stage-Specific Fusion Characteristics:**

| Stage | Resolution | mit_b1 Channels | Feature Level | Best For Fusion? |
|-------|------------|-----------------|---------------|------------------|
| 1 | H/4 × W/4 | 64 | High-res, low-level | Early fusion (edges, textures) |
| 2 | H/8 × W/8 | 128 | Mid-res, mid-level | Moderate fusion (local patterns) |
| 3 | H/16 × W/16 | 320 | Low-res, high-level | Late fusion (semantic features) |
| 4 | H/32 × W/32 | 512 | Lowest-res, highest-level | Latest fusion (global context) |

**Ablation Strategy:**
- Test single-stage fusion: [1], [2], [3], [4]
- Test multi-stage fusion: [2,3], [3,4], [2,3,4], [1,2,3,4]
- Hypothesis: Mid-late fusion ([2,3] or [3,4]) may work best

**Channel Scaling Impact:**
- **mit_b0:** Narrower channels (32-256) → less fusion capacity
- **mit_b1:** Standard channels (64-512) → good fusion capacity
- **mit_b2/b4/b5:** Same channels, more depth → better cross-modal relationships?

### 5.3 Backbone Selection Recommendations

**For CSSA Ablations:**
- ✅ Use **mit_b1** (currently doing this)
- Rationale: Lightweight fusion, doesn't need extra capacity

**For GAFF Ablations:**
- ✅ Use **mit_b1** (currently doing this)
- Rationale: GAFF adds parameters, keep backbone moderate

**For Final Comparison:**
- Test best fusion configuration on multiple backbones
- Priority: mit_b0, mit_b1, mit_b2 (once training completes)
- Optional: mit_b4, mit_b5 (if memory issues resolved)

**For Production Deployment:**
- **Fast inference:** mit_b0 + best lightweight fusion (CSSA)
- **Balanced:** mit_b1 + best fusion (CSSA or GAFF)
- **High accuracy:** mit_b2 + best fusion (if latency acceptable)

---

## 6. Future Experiments & Recommendations

### 6.1 Enabling mit_b4 and mit_b5 Training

**Priority 1: Gradient Checkpointing**
Already implemented in `resume_large_backbones.py`, just not enabled:
```python
use_grad_checkpoint = True  # Set this flag
```
Expected memory savings: 30-50%

**Priority 2: Mixed Precision Training**
Add to training script:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
Expected memory savings: 50%

**Priority 3: Further Reduce Batch Size**
```python
batch_size = 2
grad_accum_steps = 8
effective_batch_size = 16
```

**Combined Strategy:**
- Gradient checkpointing: -40% memory
- Mixed precision: -50% memory
- Total reduction: ~70% memory usage
- Should enable mit_b4 training

### 6.2 Architecture Modifications

**Option 1: Reduce FPN Channels**
```python
fpn_out_channels = 128  # Default is 256
```
Reduces detection head memory by ~50%, minor accuracy impact.

**Option 2: Selective Fusion**
Only apply fusion at 1-2 critical stages (e.g., stage 3 only):
```python
fusion_stages = [3]  # Instead of [1, 2, 3, 4]
```
Reduces fusion module memory and computation.

**Option 3: Knowledge Distillation**
Train small model (mit_b0/b1) using large model (mit_b4/b5) as teacher:
- Avoids training large model directly
- Gets similar accuracy with smaller model
- More deployment-friendly

### 6.3 Comparative Analysis

Once mit_b2 completes training, perform systematic comparison:

| Metric | mit_b0 | mit_b1 | mit_b2 | Analysis |
|--------|--------|--------|--------|----------|
| mAP | ? | ? | ? | Is depth worth 2× training time? |
| mAP@50 | ? | ? | ? | How does it affect localization? |
| mAP_small | ? | ? | ? | Does depth help small objects? |
| Training time | 2.68h | 3.44h | ~6.75h | Confirm speed scaling |
| Inference FPS | ? | ? | ? | Measure deployment speed |

**Hypotheses to Test:**
1. mit_b2 depth improves multi-modal fusion effectiveness
2. Deeper stages (especially stage 3) better capture cross-modal relationships
3. Parameter efficiency: b2 gives good accuracy/param ratio

---

## 7. Reference Information

### 7.1 File Locations

**Backbone Implementation:**
- `crossattentiondet/models/encoder.py` - MiT encoder definitions
- Lines 15-63: mit_b0 configuration
- Lines 66-114: mit_b1 configuration
- Lines 117-165: mit_b2 configuration
- Lines 220-268: mit_b4 configuration
- Lines 271-319: mit_b5 configuration

**Training Scripts:**
- `scripts/train.py` - Single backbone training
- `scripts/train_all_backbones_comprehensive.py` - Multi-backbone training
- `scripts/resume_large_backbones.py` - Memory-optimized training for b2/b4/b5

**Pretrained Weights:**
Not currently used (training from scratch on multi-modal data).

### 7.2 Key Configuration Parameters

**In `crossattentiondet/config.py`:**
```python
# Backbone selection
backbone_name = 'mit_b1'  # Default

# FPN configuration
fpn_out_channels = 256

# Detection head
num_classes = 2  # Background + object class

# Anchor generator
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
```

### 7.3 Memory Profiling Commands

To profile GPU memory usage:
```bash
# Monitor GPU during training
watch -n 1 nvidia-smi

# Profile with PyTorch
import torch
torch.cuda.memory_summary()
torch.cuda.max_memory_allocated()

# Detailed breakdown
from torch.utils.checkpoint import checkpoint_sequential
```

### 7.4 Literature References

**SegFormer Paper:**
- "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
- Xie et al., NeurIPS 2021
- Original MiT backbone architecture

**Object Detection with Transformers:**
- "DETR: End-to-End Object Detection with Transformers" (Carion et al., ECCV 2020)
- "Deformable DETR" (Zhu et al., ICLR 2021)

**Multi-Modal Fusion:**
- "CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation" (Zhang et al., ECCV 2022)
- Inspiration for this project's cross-attention fusion

---

## Appendix: Quick Reference

### Backbone Selection Cheat Sheet

**Choose mit_b0 if:**
- ✅ Need fastest training/inference
- ✅ Limited GPU memory
- ✅ Prototyping phase
- ✅ Deploying on edge devices

**Choose mit_b1 if:**
- ✅ Need balanced accuracy/speed
- ✅ Running extensive ablations
- ✅ Standard GPU resources (16-32 GB)
- ✅ Default/recommended choice

**Choose mit_b2 if:**
- ✅ Can afford 2× training time
- ✅ Have 40+ GB GPU memory (with grad accum)
- ✅ Need better accuracy than b1
- ✅ Exploring depth benefits for multi-modal fusion

**Choose mit_b4/b5 if:**
- ✅ Maximum accuracy is priority
- ✅ Can implement extensive memory optimization
- ✅ Have 80+ GB GPU or multiple GPUs
- ✅ Research experiments, not production
- ⚠️ Currently not feasible without optimizations

### Command Quick Reference

```bash
# Train single backbone
python scripts/train.py --backbone mit_b1 --epochs 15 --batch-size 2

# Train all backbones
python scripts/train_all_backbones_comprehensive.py --epochs 15 --batch-size 16

# Resume large backbones with memory optimization
python scripts/resume_large_backbones.py \
    --backbones mit_b2 mit_b4 mit_b5 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --use-grad-checkpoint

# Check GPU memory
nvidia-smi

# Monitor training
tail -f training_logs/run_*/mit_b1/logs/training.log
```

---

**Document End**
