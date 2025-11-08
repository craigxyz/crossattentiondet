# CSSA Ablation Study Guide
**Channel Switching and Spatial Attention**

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Total Experiments:** 11 (3 complete, 8 pending)

---

## Overview

The CSSA (Channel Switching and Spatial Attention) ablation study systematically explores a lightweight multi-modal fusion mechanism for combining RGB and auxiliary (thermal + event) features in object detection.

**Key Characteristics:**
- **Lightweight:** Minimal parameter overhead
- **Channel-level:** Selective channel replacement based on attention scores
- **Spatial weighting:** Final fusion uses spatial attention
- **Efficient:** Fast inference, low memory footprint

**Ablation Goals:**
1. Identify which encoder stages benefit most from CSSA fusion
2. Determine optimal channel switching threshold
3. Understand the impact of fusion placement on detection accuracy

---

## 1. CSSA Architecture

### 1.1 Overall Structure

```
Input: RGB Features (B,C,H,W) + Auxiliary Features (B,C,H,W)

┌─────────────────────────────────────────────────────────────┐
│                     CSSA Fusion Module                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RGB (B,C,H,W) ─────┬─→ ECA_RGB ──→ Attn_RGB (B,C,1,1)    │
│                     │                      ↓                │
│                     │              Channel Switching        │
│                     │             (threshold-based)         │
│                     │                      ↓                │
│                     │              RGB_switched (B,C,H,W)  │
│                     │                      ↓                │
│                     │              ┌──────────────┐         │
│                     └──────────────┤              │         │
│                                    │   Spatial    │         │
│  Aux (B,C,H,W) ─────┬─→ ECA_Aux ──┤  Attention   │──→ Out │
│                     │              │              │         │
│                     │              └──────────────┘         │
│                     │                      ↑                │
│                     │              Aux_switched (B,C,H,W)  │
│                     │                      ↑                │
│                     │              Channel Switching        │
│                     │             (threshold-based)         │
│                     │                      ↑                │
│                     └─→ ECA_Aux ──→ Attn_Aux (B,C,1,1)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Output: Fused Features (B,C,H,W)
```

### 1.2 Component Breakdown

#### **Component 1: ECABlock (Efficient Channel Attention)**

**Location:** `crossattentiondet/ablations/fusion/cssa.py:14-37`

**Purpose:** Generate channel-wise attention weights for each modality

**Architecture:**
```python
Input: (B, C, H, W)
  ↓
Global Average Pooling (AdaptiveAvgPool2d)
  ↓
Channel weights: (B, C, 1, 1)
  ↓
Transpose to (B, 1, C, 1) for 1D conv
  ↓
1D Convolution (kernel_size=k, padding=(k-1)//2)
  ↓
Transpose back to (B, C, 1, 1)
  ↓
Sigmoid activation
  ↓
Output: Attention weights (B, C, 1, 1), range [0, 1]
```

**Key Feature - Adaptive Kernel Size:**
```python
k = get_kernel_size(C, gamma=2, b=1)
# Adaptively determines 1D conv kernel based on channel dimension
# More channels → larger receptive field
# Example: C=64 → k=3, C=128 → k=5, C=512 → k=9
```

**Parameters:**
- Input C channels → 1D Conv(kernel=k) → C output channels
- Total params: C × k (very lightweight)
- Example: C=512, k=9 → 4,608 parameters

#### **Component 2: ChannelSwitching**

**Location:** `crossattentiondet/ablations/fusion/cssa.py:40-70`

**Purpose:** Selectively swap channels between RGB and Aux based on confidence

**Algorithm:**
```python
Input: rgb_feat (B,C,H,W), aux_feat (B,C,H,W), attn_rgb (B,C,1,1), attn_aux (B,C,1,1)

For each channel c:
  if attn_rgb[c] > threshold:
    # RGB is confident, keep RGB channel
    rgb_out[c] = rgb_feat[c]
    aux_out[c] = aux_feat[c]  # Aux stays as is
  else if attn_aux[c] > threshold:
    # Aux is confident, swap channels
    rgb_out[c] = aux_feat[c]  # Replace RGB with Aux
    aux_out[c] = rgb_feat[c]  # Replace Aux with RGB
  else:
    # Both low confidence, keep original
    rgb_out[c] = rgb_feat[c]
    aux_out[c] = aux_feat[c]

Output: rgb_switched (B,C,H,W), aux_switched (B,C,H,W)
```

**Threshold Behavior:**
- **threshold=0.3:** More aggressive switching (low bar for "confident")
- **threshold=0.5:** Balanced switching (default)
- **threshold=0.7:** Conservative switching (high bar for "confident")

**Parameters:** None (threshold is a hyperparameter, not learnable)

#### **Component 3: SpatialAttention**

**Location:** `crossattentiondet/ablations/fusion/cssa.py:73-103`

**Purpose:** Weight spatial locations based on combined feature importance

**Architecture:**
```python
Input: rgb (B,C,H,W), aux (B,C,H,W)

Concatenate along channel dimension: (B, 2C, H, W)
  ↓
Channel-wise Average Pooling: (B, 1, H, W)
Channel-wise Max Pooling: (B, 1, H, W)
  ↓
Concatenate: (B, 2, H, W)
  ↓
Conv2d(2, 1, kernel_size=7, padding=3)
  ↓
Sigmoid: (B, 1, H, W)  # Spatial attention map
  ↓
Multiply attention map with input:
  fused = attn_map * rgb + (1 - attn_map) * aux
  ↓
Output: (B, C, H, W)
```

**Alternative fusion** (controlled by `use_conv_fusion`):
```python
# Instead of weighted sum, use convolution
Concatenate: (B, 2C, H, W)
  ↓
Conv2d(2C, C, kernel_size=1) + BatchNorm + ReLU
  ↓
Output: (B, C, H, W)
```

**Parameters:**
- 7×7 conv: 2 × 1 × 7 × 7 + 1 = 99 parameters (spatial attention)
- 1×1 conv (if used): 2C × C + C = 2C² + C parameters (conv fusion)

### 1.3 Complete CSSA Module

**Location:** `crossattentiondet/ablations/fusion/cssa.py:106-143`

**Full Forward Pass:**
```python
def forward(rgb, aux):
    # Step 1: Generate channel attention for each modality
    attn_rgb = eca_rgb(rgb)     # (B, C, 1, 1)
    attn_aux = eca_aux(aux)     # (B, C, 1, 1)

    # Step 2: Channel switching based on threshold
    rgb_switched, aux_switched = channel_switching(rgb, aux, attn_rgb, attn_aux)

    # Step 3: Spatial attention for final fusion
    fused = spatial_attention(rgb_switched, aux_switched)

    return fused  # (B, C, H, W)
```

**Total Parameters (per stage for C channels):**
- ECA_RGB: C × k_rgb
- ECA_Aux: C × k_aux
- Channel Switching: 0 (no learnable params)
- Spatial Attention: 99 or (2C² + C)
- **Total: ~2Ck + 99** (lightweight) or **2C² + 2Ck + C** (with conv fusion)

**Example for mit_b1 Stage 3 (C=320):**
- ECA: 2 × 320 × 7 = 4,480 params
- Spatial: 99 params
- **Total: 4,579 params** (0.007% of 69.5M total model params!)

---

## 2. Ablation Design

### 2.1 Two-Phase Approach

**Phase 1: Stage Selection** (7 experiments)
- **Goal:** Identify which encoder stages benefit most from CSSA fusion
- **Fixed parameters:** threshold=0.5, kernel=3 (default)
- **Variable:** Which stages apply CSSA vs. identity passthrough

**Phase 2: Threshold Sensitivity** (4 experiments)
- **Goal:** Fine-tune channel switching threshold for top configurations
- **Fixed parameters:** Best 2 stage configurations from Phase 1, kernel=3
- **Variable:** threshold ∈ {0.3, 0.5, 0.7}

### 2.2 Phase 1: Stage Selection

**Rationale:**
- Different stages capture different semantic levels
- Early stages (1, 2): Low-level features (edges, textures)
- Late stages (3, 4): High-level features (objects, semantics)
- Multi-modal fusion may work better at certain levels

**Experiment Matrix:**

| Exp ID | Stages | Code | Hypothesis | Status |
|--------|--------|------|------------|--------|
| exp_001 | [1] | s1_t0.5 | Early fusion helps low-level alignment | ✅ Complete |
| exp_002 | [2] | s2_t0.5 | Mid-early fusion balances detail & semantics | ✅ Complete |
| exp_003 | [3] | s3_t0.5 | Mid-late fusion for semantic correlation | ✅ Complete |
| exp_004 | [4] | s4_t0.5 | Late fusion for high-level object features | ⏸️ Pending |
| exp_005 | [2,3] | s23_t0.5 | Combined mid-level fusion | ⏸️ Pending |
| exp_006 | [3,4] | s34_t0.5 | Combined late fusion | ⏸️ Pending |
| exp_007 | [1,2,3,4] | s1234_t0.5 | All-stage fusion (maximum interaction) | ⏸️ Pending |

**Stage Characteristics (mit_b1):**

| Stage | Resolution | Channels | Semantic Level | Typical Features | Fusion Hypothesis |
|-------|------------|----------|----------------|------------------|-------------------|
| 1 | H/4 × W/4 | 64 | Very Low | Edges, colors, gradients | Align textures across modalities |
| 2 | H/8 × W/8 | 128 | Low | Simple patterns, corners | Combine local patterns |
| 3 | H/16 × W/16 | 320 | High | Parts, semantic regions | Fuse object parts (likely best) |
| 4 | H/32 × W/32 | 512 | Very High | Whole objects, global context | Integrate global semantics |

**Expected Results:**
- **Best single stage:** Likely stage 3 (mid-late, high-level semantics)
- **Best multi-stage:** Likely [2,3] or [3,4] (avoid stage 1 overhead, maximize semantic fusion)
- **Worst:** Likely stage 1 alone (too low-level for object detection)
- **All stages [1,2,3,4]:** May overfit or add unnecessary computation

### 2.3 Phase 2: Threshold Sensitivity

**Rationale:**
- Threshold controls switching aggressiveness
- Too low (0.3): Over-switching, may disrupt features
- Too high (0.7): Under-switching, limited modality interaction
- Optimal depends on dataset and modality complementarity

**Experiment Matrix** (TBD based on Phase 1 results):

Assuming top 2 configs are **s3_t0.5** and **s23_t0.5**:

| Exp ID | Stages | Threshold | Code | Hypothesis |
|--------|--------|-----------|------|------------|
| exp_008 | [3] | 0.3 | s3_t0.3 | More aggressive switching improves fusion |
| exp_009 | [3] | 0.7 | s3_t0.7 | Conservative switching is safer |
| exp_010 | [2,3] | 0.3 | s23_t0.3 | Multi-stage benefits from more switching |
| exp_011 | [2,3] | 0.7 | s23_t0.7 | Multi-stage with conservative switching |

**Threshold Impact Analysis:**

| Threshold | Switching Behavior | Expected Pros | Expected Cons |
|-----------|-------------------|---------------|---------------|
| 0.3 | Aggressive (switch often) | More cross-modal interaction | May disrupt good RGB features |
| 0.5 | Balanced (default) | Moderate interaction | May be sub-optimal |
| 0.7 | Conservative (switch rarely) | Preserves confident features | Limited modality fusion |

**Metrics to Analyze:**
- Overall mAP (does threshold improve accuracy?)
- mAP by object size (small/medium/large)
- Training stability (loss curves)
- Inference speed (should be similar across thresholds)

---

## 3. Training Configuration

### 3.1 Standard Settings

**Backbone:** mit_b1 (default, configurable)
```python
backbone_name = 'mit_b1'
total_params = 69.5M (base) + CSSA overhead
```

**Training Hyperparameters:**
```python
epochs = 25
batch_size = 2
learning_rate = 0.005
optimizer = SGD(momentum=0.9, weight_decay=1e-4)
lr_scheduler = None  # Constant learning rate
```

**Data:**
```python
data_dir = '../RGBX_Semantic_Segmentation/data/images'
labels_dir = '../RGBX_Semantic_Segmentation/data/labels'
train_split = 0.8  # 80% train, 20% test
num_classes = 2  # Background + object class
```

**CSSA Parameters:**
```python
# Phase 1: Fixed
threshold = 0.5
kernel_size = 3  # Can be adaptive based on channels

# Phase 2: Variable
threshold ∈ {0.3, 0.5, 0.7}
```

### 3.2 Encoder Configuration

**CSSA Integration:**

The encoder checks `fusion_stages` to apply CSSA selectively:

```python
# Example: fusion_stages = [2, 3]

def forward(self, rgb, aux):
    # Stage 1: No fusion (identity)
    rgb_1, aux_1 = stage1(rgb), stage1(aux)

    # Stage 2: CSSA fusion
    if 2 in fusion_stages:
        fused_2 = cssa_stage2(rgb_2, aux_2)  # Apply CSSA
        rgb_2, aux_2 = fused_2, fused_2  # Both streams get fused features

    # Stage 3: CSSA fusion
    if 3 in fusion_stages:
        fused_3 = cssa_stage3(rgb_3, aux_3)
        rgb_3, aux_3 = fused_3, fused_3

    # Stage 4: No fusion (identity)
    rgb_4, aux_4 = stage4(rgb_3), stage4(aux_3)

    return [fused_1/rgb_1, fused_2/rgb_2, fused_3, fused_4]
```

**Key Decision:** After fusion, both RGB and Aux streams carry the same fused features (symmetric design).

### 3.3 Evaluation Metrics

**COCO Detection Metrics:**
```python
metrics = {
    'mAP': 'Mean Average Precision (IoU 0.5:0.95)',
    'mAP@50': 'AP at IoU threshold 0.5',
    'mAP@75': 'AP at IoU threshold 0.75',
    'mAP_small': 'AP for small objects (area < 32²)',
    'mAP_medium': 'AP for medium objects (32² < area < 96²)',
    'mAP_large': 'AP for large objects (area > 96²)'
}
```

**Additional Metrics:**
- Training loss (per epoch)
- Validation loss
- Training time (hours)
- Inference time (FPS)
- Model parameters (millions)
- GPU memory usage (GB)

---

## 4. Results & Analysis

### 4.1 Current Status

**Completed Experiments:** 3/11 (27.3%)

| Exp ID | Config | Status | Best mAP | Best mAP@50 | Epochs | Notes |
|--------|--------|--------|----------|-------------|--------|-------|
| exp_001 | s1_t0.5 | ✅ Complete | TBD | TBD | 25 | Results in `results/cssa_ablations/exp_001_*/` |
| exp_002 | s2_t0.5 | ✅ Complete | TBD | TBD | 25 | Results in `results/cssa_ablations/exp_002_*/` |
| exp_003 | s3_t0.5 | ✅ Complete | TBD | TBD | 25 | Results in `results/cssa_ablations/exp_003_*/` |

**Pending Experiments:** 8/11

### 4.2 Results Analysis (Placeholder)

**To be filled after Phase 1 completion:**

**Stage-wise Performance:**
```
Stage Configuration | mAP | mAP@50 | mAP@75 | Params | Time (hrs) | Ranking
--------------------|-----|--------|--------|--------|------------|--------
[1]                 | ?   | ?      | ?      | ?      | ?          | ?
[2]                 | ?   | ?      | ?      | ?      | ?          | ?
[3]                 | ?   | ?      | ?      | ?      | ?          | ?
[4]                 | ?   | ?      | ?      | ?      | ?          | ?
[2,3]               | ?   | ?      | ?      | ?      | ?          | ?
[3,4]               | ?   | ?      | ?      | ?      | ?          | ?
[1,2,3,4]           | ?   | ?      | ?      | ?      | ?          | ?
```

**Threshold Sensitivity** (Phase 2):
```
Config | Threshold | mAP | mAP@50 | Improvement vs. 0.5 | Ranking
-------|-----------|-----|--------|---------------------|--------
Top #1 | 0.3       | ?   | ?      | ?                   | ?
Top #1 | 0.5       | ?   | ?      | Baseline            | ?
Top #1 | 0.7       | ?   | ?      | ?                   | ?
Top #2 | 0.3       | ?   | ?      | ?                   | ?
Top #2 | 0.5       | ?   | ?      | Baseline            | ?
Top #2 | 0.7       | ?   | ?      | ?                   | ?
```

### 4.3 Visualizations (To be generated)

**1. Stage Performance Comparison:**
- Bar chart: mAP by stage configuration
- Line plot: mAP@{50, 75} by stage

**2. Threshold Sensitivity:**
- Line plot: mAP vs. threshold for top 2 configs
- Heatmap: Switching frequency by channel and threshold

**3. Training Curves:**
- Loss curves for all 11 experiments
- Validation mAP over epochs

**4. Computational Cost:**
- Training time vs. mAP (efficiency analysis)
- Parameters vs. mAP (parameter efficiency)

---

## 5. Implementation Details

### 5.1 Running CSSA Ablations

**Script:** `crossattentiondet/ablations/scripts/run_cssa_ablations.py`

**Usage:**
```bash
# Run complete Phase 1 (all 7 stage configurations)
python crossattentiondet/ablations/scripts/run_cssa_ablations.py \
    --phase 1 \
    --backbone mit_b1 \
    --epochs 25 \
    --batch-size 2

# Run specific experiment
python crossattentiondet/ablations/scripts/run_cssa_ablations.py \
    --exp-id exp_004 \
    --backbone mit_b1 \
    --epochs 25

# Run Phase 2 after Phase 1 completes
python crossattentiondet/ablations/scripts/run_cssa_ablations.py \
    --phase 2 \
    --top-configs s3 s23 \
    --thresholds 0.3 0.7 \
    --epochs 25
```

### 5.2 Code Locations

**CSSA Module:**
```
crossattentiondet/ablations/fusion/cssa.py
├── ECABlock (lines 14-37)
├── ChannelSwitching (lines 40-70)
├── SpatialAttention (lines 73-103)
└── CSSAFusion (lines 106-143)
```

**Encoder Integration:**
```
crossattentiondet/ablations/encoder_cssa.py
└── CMXEncoder_CSSA (modified to accept fusion_stages parameter)
```

**Ablation Framework:**
```
crossattentiondet/ablations/scripts/run_cssa_ablations.py
├── Phase 1: Stage selection (lines 50-120)
├── Phase 2: Threshold tuning (lines 123-180)
└── Experiment management (logging, checkpointing)
```

### 5.3 Output Structure

**Results Directory:**
```
results/cssa_ablations/
├── exp_001_s1_t0.5/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── epoch_*.pth
│   ├── logs/
│   │   ├── training_info.json
│   │   ├── epoch_metrics.csv
│   │   └── eval_results.json
│   └── config.json
├── exp_002_s2_t0.5/
│   └── ...
└── summary_phase1.json
```

**Checkpoints:**
- `best_model.pth`: Model with best validation mAP
- `epoch_*.pth`: Checkpoints every epoch

**Logs:**
- `training_info.json`: Config, status, parameters
- `epoch_metrics.csv`: Per-epoch loss, mAP, time
- `eval_results.json`: Final COCO eval results
- `config.json`: Experiment configuration

---

## 6. Analysis Guidelines

### 6.1 Phase 1 Analysis

**After all 7 experiments complete:**

1. **Rank by mAP:**
   - Identify top 2-3 stage configurations
   - Check if hypothesis (stage 3 or [2,3] best) holds

2. **Analyze by object size:**
   - Do early stages (1,2) help small objects?
   - Do late stages (3,4) help large objects?

3. **Computational cost:**
   - Is all-stage fusion [1,2,3,4] worth the overhead?
   - Best accuracy/cost trade-off?

4. **Training stability:**
   - Which configs converge fastest?
   - Any overfitting in multi-stage configs?

5. **Select top 2 for Phase 2:**
   - Highest mAP
   - Best accuracy/cost if close

### 6.2 Phase 2 Analysis

**After threshold experiments:**

1. **Threshold sensitivity:**
   - Is performance sensitive to threshold?
   - Does optimal threshold differ by stage config?

2. **Switching statistics:**
   - Count actual channel swaps during inference
   - Visualize which channels switch most

3. **Modality importance:**
   - Does low threshold (more switching) help?
   - → Suggests aux modality is valuable
   - Does high threshold (less switching) help?
   - → Suggests RGB is already good, fusion should be conservative

4. **Final recommendation:**
   - Best single configuration (stage + threshold)
   - Expected mAP improvement vs. baseline

### 6.3 Comparison with Baseline

**Baseline (no fusion):**
- Use mit_b1 trained without any fusion
- Compare mAP, training time, parameters

**CSSA vs. Baseline:**
```
Metric          | Baseline | Best CSSA | Improvement | Cost
----------------|----------|-----------|-------------|-----
mAP             | ?        | ?         | ?           | +~5K params
mAP@50          | ?        | ?         | ?           | +~1% train time
mAP@75          | ?        | ?         | ?           |
Params (M)      | 69.5     | ~69.51    | +0.01       |
Time (hrs)      | ?        | ?         | +~5%        |
```

**Questions to Answer:**
- Is CSSA worth the complexity for the accuracy gain?
- Does it outperform simple late concatenation?
- How does it compare to GAFF (see GAFF_ABLATION_GUIDE.md)?

---

## 7. Troubleshooting & FAQs

### 7.1 Common Issues

**Q: Experiment fails with "dimension mismatch"**
A: Check that fusion_stages contains valid stage numbers [1,2,3,4]. Make sure encoder stages match the fusion module stages.

**Q: Training is very slow**
A: CSSA is lightweight, slowdown likely from data loading or evaluation. Try:
- Increase `num_workers` for data loading
- Reduce evaluation frequency
- Check GPU utilization with `nvidia-smi`

**Q: mAP is lower than baseline**
A: Possible causes:
- Fusion stages are not optimal (try different configs)
- Threshold is too aggressive or conservative
- Need more training epochs (25 may not be enough)
- Learning rate may need tuning

**Q: All stage configurations give similar results**
A: Suggests fusion placement doesn't matter much, which means:
- Dataset may not benefit from multi-modal fusion
- CSSA design may need improvement
- Baseline RGB features already very strong

### 7.2 Debug Mode

**Enable detailed logging:**
```python
python crossattentiondet/ablations/scripts/run_cssa_ablations.py \
    --debug \
    --log-level DEBUG
```

**Visualize attention maps:**
```python
# In CSSA module, save attention weights during forward pass
torch.save(attn_rgb, 'debug/attn_rgb.pt')
torch.save(attn_aux, 'debug/attn_aux.pt')

# Visualize
import matplotlib.pyplot as plt
attn = torch.load('debug/attn_rgb.pt').squeeze().cpu().numpy()
plt.bar(range(len(attn)), attn)
plt.xlabel('Channel')
plt.ylabel('Attention Weight')
plt.show()
```

### 7.3 Extending CSSA

**Add new hyperparameters:**

1. **Variable kernel size:**
   ```python
   # In run_cssa_ablations.py
   for kernel in [3, 5, 7]:
       run_experiment(stages=[3], threshold=0.5, kernel=kernel)
   ```

2. **Different fusion strategies:**
   ```python
   # In CSSAFusion, add parameter for fusion type
   spatial_attention = SpatialAttention(use_conv_fusion=True/False)
   ```

3. **Asymmetric switching:**
   ```python
   # Use different thresholds for RGB and Aux
   threshold_rgb = 0.7  # Conservative for RGB
   threshold_aux = 0.3  # Aggressive for Aux
   ```

---

## 8. Future Work & Extensions

### 8.1 Potential Improvements

1. **Learnable threshold:**
   - Make threshold a learnable parameter per channel
   - Initialize at 0.5, train with gradients

2. **Soft switching:**
   - Instead of hard threshold, use weighted combination
   - `alpha = sigmoid(attn_rgb - attn_aux)`
   - `fused = alpha * rgb + (1-alpha) * aux`

3. **Stage-adaptive fusion:**
   - Different fusion strategies per stage
   - Early: simple average, Late: complex attention

4. **Multi-scale fusion:**
   - Fuse features across adjacent stages
   - E.g., use stage 2 features to guide stage 3 fusion

### 8.2 Comparison Studies

**CSSA vs. Other Fusion Methods:**
1. Simple concatenation + 1×1 conv
2. Element-wise addition
3. GAFF (see GAFF_ABLATION_GUIDE.md)
4. Cross-attention (full transformer)

**Create comparison table:**
```
Method      | mAP  | Params | Time | Complexity
------------|------|--------|------|------------
Concat+Conv | ?    | +2C²   | ?    | Low
Add         | ?    | 0      | ?    | Very Low
CSSA        | ?    | +2Ck   | ?    | Low
GAFF        | ?    | +HIGH  | ?    | Medium-High
CrossAttn   | ?    | +VERY  | ?    | High
```

### 8.3 Application to Other Backbones

**Test CSSA on different backbones:**
```bash
# Lightweight
python run_cssa_ablations.py --backbone mit_b0 --best-config s3_t0.5

# Deeper (if memory allows)
python run_cssa_ablations.py --backbone mit_b2 --best-config s3_t0.5
```

**Hypothesis:** CSSA benefits may scale with backbone capacity.

---

## Appendix: Quick Reference

### Experiment Naming Convention
```
exp_<XXX>_s<stage_label>_t<threshold>

Examples:
- exp_001_s1_t0.5   → Stage 1, threshold 0.5
- exp_005_s23_t0.5  → Stages 2+3, threshold 0.5
- exp_008_s3_t0.3   → Stage 3, threshold 0.3

Stage labels:
s1    = [1]
s2    = [2]
s3    = [3]
s4    = [4]
s23   = [2, 3]
s34   = [3, 4]
s234  = [2, 3, 4]
s1234 = [1, 2, 3, 4]
```

### Commands Cheat Sheet
```bash
# Run Phase 1
python crossattentiondet/ablations/scripts/run_cssa_ablations.py --phase 1

# Run specific experiment
python crossattentiondet/ablations/scripts/run_cssa_ablations.py --exp-id exp_004

# Check results
cat results/cssa_ablations/exp_001_*/logs/eval_results.json

# Monitor training
tail -f results/cssa_ablations/exp_001_*/logs/training.log

# Visualize results
python crossattentiondet/ablations/scripts/visualize_cssa_results.py
```

### Key Files
```
crossattentiondet/ablations/fusion/cssa.py              # CSSA module
crossattentiondet/ablations/encoder_cssa.py             # Encoder with CSSA
crossattentiondet/ablations/scripts/run_cssa_ablations.py  # Ablation runner
results/cssa_ablations/                                 # Results directory
```

---

**Document End**
