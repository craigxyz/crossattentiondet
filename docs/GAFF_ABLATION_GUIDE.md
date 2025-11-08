# GAFF Ablation Study Guide
**Guided Attentive Feature Fusion**

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Total Experiments:** 32 (4 complete, 28 pending)

---

## Overview

The GAFF (Guided Attentive Feature Fusion) ablation study systematically explores an attention-based multi-modal fusion mechanism for combining RGB and auxiliary (thermal + event) features in object detection. GAFF is based on the WACV 2021 paper "Guided Attentive Feature Fusion for Multispectral Pedestrian Detection" by Zhang et al.

**Key Characteristics:**
- **Attention-rich:** Combines both intra-modality and inter-modality attention
- **Guided fusion:** Uses learned gates to weight cross-modal information flow
- **Moderate complexity:** More sophisticated than CSSA, higher parameter count
- **Proven effectiveness:** Published approach with strong empirical results

**Ablation Goals:**
1. Identify which encoder stages benefit most from GAFF fusion
2. Determine optimal SE reduction ratio for channel attention
3. Evaluate shared vs. separate inter-modality attention pathways
4. Compare bottleneck vs. direct merge strategies

---

## 1. GAFF Architecture

### 1.1 Overall Structure

```
Input: RGB Features (B,C,H,W) + Auxiliary Features (B,C,H,W)

┌─────────────────────────────────────────────────────────────┐
│                     GAFF Fusion Module                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Intra-modality Attention (SE Blocks)              │
│  ────────────────────────────────────────────               │
│  RGB (B,C,H,W) ──→ SEBlock_RGB ──→ x̂_rgb (B,C,H,W)        │
│  Aux (B,C,H,W) ──→ SEBlock_Aux ──→ x̂_aux (B,C,H,W)        │
│                                                             │
│  Step 2: Inter-modality Attention (Cross-Attention)        │
│  ─────────────────────────────────────────────              │
│  Concat[x_rgb, x_aux] (B, 2C, H, W)                        │
│         ↓                                                   │
│  ┌──────────────────────────────┐                          │
│  │ InterModalityAttention       │                          │
│  │  - Shared: 1 conv → split    │                          │
│  │  - Separate: 2 convs         │                          │
│  └──────────────────────────────┘                          │
│         ↓                    ↓                              │
│  w_rgb←aux (B,C,H,W)  w_aux←rgb (B,C,H,W)                 │
│                                                             │
│  Step 3: Guided Fusion                                     │
│  ──────────────────                                         │
│  x̂_rgb_guided = x̂_rgb + w_rgb←aux * x_aux                 │
│  x̂_aux_guided = x̂_aux + w_aux←rgb * x_rgb                 │
│                                                             │
│  Step 4: Merge                                             │
│  ──────────                                                 │
│  Concat[x̂_rgb_guided, x̂_aux_guided] (B, 2C, H, W)         │
│         ↓                                                   │
│  Option A (merge_bottleneck=False):                        │
│    Conv2d(2C → out_C) + BN                                 │
│                                                             │
│  Option B (merge_bottleneck=True):                         │
│    Conv2d(2C → C) + BN + ReLU                              │
│    Conv2d(C → out_C) + BN                                  │
│         ↓                                                   │
│  Output: Fused Features (B, out_C, H, W)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Breakdown

#### **Component 1: SEBlock (Squeeze-and-Excitation)**

**Location:** `crossattentiondet/ablations/fusion/gaff.py:22-65`

**Purpose:** Generate channel-wise attention weights for intra-modality feature refinement

**Architecture:**
```python
Input: (B, C, H, W)
  ↓
Global Average Pooling (AdaptiveAvgPool2d)
  ↓
Channel descriptor: (B, C, 1, 1) → (B, C)
  ↓
FC1: Linear(C → C/reduction)
  ↓
ReLU activation
  ↓
FC2: Linear(C/reduction → C)
  ↓
Sigmoid activation
  ↓
Attention weights: (B, C, 1, 1), range [0, 1]
  ↓
Multiply with input: x * attention_weights
  ↓
Output: Attention-weighted features (B, C, H, W)
```

**Parameters (per SE block):**
- FC1: C × (C/reduction) + 0 bias = C²/reduction
- FC2: (C/reduction) × C + 0 bias = C²/reduction
- **Total: 2C²/reduction**

**Hyperparameter: SE Reduction Ratio**
- `reduction=4`: More parameters, richer representation
  - Example (C=320): 2×320²/4 = 51,200 params
- `reduction=8`: Fewer parameters, stronger bottleneck
  - Example (C=320): 2×320²/8 = 25,600 params

**Design Rationale:**
- Higher reduction (8) → More compression, faster, fewer params
- Lower reduction (4) → Less compression, richer bottleneck, more params

#### **Component 2: InterModalityAttention (Cross-Attention)**

**Location:** `crossattentiondet/ablations/fusion/gaff.py:67-121`

**Purpose:** Compute cross-attention weights between RGB and auxiliary modalities

**Architecture (Shared Mode):**
```python
Input: x_rgb (B,C,H,W), x_aux (B,C,H,W)
  ↓
Concatenate: (B, 2C, H, W)
  ↓
Conv2d(2C → 2C, kernel_size=1)
  ↓
Split into two halves: (B, C, H, W) each
  ↓
Sigmoid activation
  ↓
Output: w_rgb←aux (B,C,H,W), w_aux←rgb (B,C,H,W)
```

**Architecture (Separate Mode):**
```python
Input: x_rgb (B,C,H,W), x_aux (B,C,H,W)
  ↓
Concatenate: (B, 2C, H, W)
  ↓                ↓
Conv_rgb(2C → C)   Conv_aux(2C → C)
  ↓                ↓
Sigmoid            Sigmoid
  ↓                ↓
w_rgb←aux          w_aux←rgb
```

**Parameters:**
- **Shared mode (`inter_shared=True`):**
  - Single conv: 2C × 2C × 1 × 1 = 4C²
  - **Total: 4C²**
- **Separate mode (`inter_shared=False`):**
  - Conv_rgb: 2C × C × 1 × 1 = 2C²
  - Conv_aux: 2C × C × 1 × 1 = 2C²
  - **Total: 4C²**

**Note:** Both modes have identical parameter counts! The difference is in expressiveness:
- Shared: Forces symmetry between RGB→Aux and Aux→RGB attention
- Separate: Allows asymmetric attention (RGB and Aux can have different importance)

#### **Component 3: Guided Fusion**

**Location:** `crossattentiondet/ablations/fusion/gaff.py:186-193`

**Purpose:** Combine intra-modality refined features with cross-modality information

**Algorithm:**
```python
# After SE blocks
x_rgb_se = SE_rgb(x_rgb)  # Intra-modality refinement
x_aux_se = SE_aux(x_aux)

# After inter-modality attention
w_rgb_from_aux, w_aux_from_rgb = InterAttn(x_rgb, x_aux)

# Guided fusion: Add cross-modal information
x_rgb_guided = x_rgb_se + w_rgb_from_aux * x_aux
x_aux_guided = x_aux_se + w_aux_from_rgb * x_rgb
```

**Key Insight:** This is a residual-style connection where:
- SE provides the main path (modality-specific refinement)
- Cross-attention provides the additive residual (cross-modal enhancement)

**Parameters:** None (uses weights from previous components)

#### **Component 4: Merge Layer**

**Location:** `crossattentiondet/ablations/fusion/gaff.py:163-172, 196-209`

**Purpose:** Combine guided RGB and Aux features into final fused output

**Direct Merge (`merge_bottleneck=False`):**
```python
Concatenate: (B, 2C, H, W)
  ↓
Conv2d(2C → out_channels, kernel_size=1) + BatchNorm
  ↓
Output: (B, out_channels, H, W)
```

**Parameters (Direct):**
- Conv: 2C × out_channels × 1 × 1 = 2C × out_channels
- BN: 2 × out_channels (γ, β)
- **Total: 2C × out_channels + 2 × out_channels**

**Bottleneck Merge (`merge_bottleneck=True`):**
```python
Concatenate: (B, 2C, H, W)
  ↓
Conv2d(2C → C, kernel_size=1) + BatchNorm + ReLU
  ↓
Conv2d(C → out_channels, kernel_size=1) + BatchNorm
  ↓
Output: (B, out_channels, H, W)
```

**Parameters (Bottleneck):**
- Conv1: 2C × C = 2C²
- BN1: 2C
- Conv2: C × out_channels
- BN2: 2 × out_channels
- **Total: 2C² + C × out_channels + 2C + 2 × out_channels**

**Design Trade-off:**
- Direct: Fewer parameters, single transformation
- Bottleneck: More parameters, richer non-linear transformation

### 1.3 Complete GAFF Module

**Location:** `crossattentiondet/ablations/fusion/gaff.py:123-220`

**Full Forward Pass:**
```python
def forward(x_rgb, x_aux):
    # Step 1: Intra-modality attention (SE)
    x_rgb_se = se_rgb(x_rgb)
    x_aux_se = se_aux(x_aux)

    # Step 2: Inter-modality attention
    w_rgb_from_aux, w_aux_from_rgb = inter_attn(x_rgb, x_aux)

    # Step 3: Guided fusion
    x_rgb_guided = x_rgb_se + w_rgb_from_aux * x_aux
    x_aux_guided = x_aux_se + w_aux_from_rgb * x_rgb

    # Step 4: Merge
    concat_feat = cat([x_rgb_guided, x_aux_guided], dim=1)

    if merge_bottleneck:
        out = merge_conv1(concat_feat)
        out = merge_bn1(out)
        out = relu(out)
        out = merge_conv2(out)
        out = merge_bn2(out)
    else:
        out = merge_conv(concat_feat)
        out = merge_bn(out)

    return out
```

**Total Parameters (assuming out_channels = C):**

**Configuration 1: reduction=4, shared=False, bottleneck=False**
- SE blocks: 2 × (2C²/4) = C²
- Inter-attention: 4C²
- Merge: 2C² + 2C
- **Total: 7C² + 2C**
- Example (C=320): 7×320² + 2×320 = 717,440 params

**Configuration 2: reduction=8, shared=True, bottleneck=True**
- SE blocks: 2 × (2C²/8) = C²/2
- Inter-attention: 4C²
- Merge: 2C² + C² + 2C + 2C = 3C² + 4C
- **Total: 7.5C² + 4C**
- Example (C=320): 7.5×320² + 4×320 = 769,280 params

**Comparison with CSSA (C=320):**
- CSSA: ~2Ck + 99 ≈ 2×320×3 + 99 = 2,019 params
- GAFF: ~7C² + 2C ≈ 717,440 params
- **GAFF is ~355× more parameters than CSSA!**

---

## 2. Ablation Design

### 2.1 Two-Phase Approach

**Phase 1: Stage Selection** (8 experiments)
- **Goal:** Identify which encoder stages benefit most from GAFF fusion
- **Fixed parameters:** SE_reduction=4, inter_shared=False, merge_bottleneck=False (default)
- **Variable:** Which stages apply GAFF vs. identity passthrough

**Phase 2: Hyperparameter Tuning** (24 experiments)
- **Goal:** Fine-tune GAFF hyperparameters for top 3 stage configurations
- **Fixed parameters:** Best 3 stage configurations from Phase 1
- **Variable:** All combinations of {SE_reduction, inter_shared, merge_bottleneck}

### 2.2 Phase 1: Stage Selection

**Rationale:**
- Different stages capture different semantic levels
- Early stages (1, 2): Low-level features (edges, textures) - H/4, H/8
- Late stages (3, 4): High-level features (objects, semantics) - H/16, H/32
- GAFF's rich attention may be most beneficial at specific semantic levels

**Experiment Matrix:**

| Exp ID | Stages | Code | Hypothesis | Status |
|--------|--------|------|------------|--------|
| exp_001 | [1] | s1_r4_is0_mb0 | Early fusion for low-level alignment | ✅ Complete |
| exp_002 | [2] | s2_r4_is0_mb0 | Mid-early fusion balances detail & semantics | ✅ Complete |
| exp_003 | [3] | s3_r4_is0_mb0 | Mid-late fusion for semantic correlation | ✅ Complete |
| exp_004 | [4] | s4_r4_is0_mb0 | Late fusion for high-level object features | ✅ Complete |
| exp_005 | [2,3] | s23_r4_is0_mb0 | Combined mid-level fusion | ⏸️ Pending |
| exp_006 | [3,4] | s34_r4_is0_mb0 | Combined late fusion | ⏸️ Pending |
| exp_007 | [2,3,4] | s234_r4_is0_mb0 | Multi-level late fusion | ⏸️ Pending |
| exp_008 | [1,2,3,4] | s1234_r4_is0_mb0 | All-stage fusion (maximum interaction) | ⏸️ Pending |

**Stage Characteristics (mit_b1):**

| Stage | Resolution | Channels | Semantic Level | GAFF Params | Fusion Hypothesis |
|-------|------------|----------|----------------|-------------|-------------------|
| 1 | H/4 × W/4 | 64 | Very Low | ~29K | Texture/color alignment across modalities |
| 2 | H/8 × W/8 | 128 | Low | ~115K | Local pattern correlation |
| 3 | H/16 × W/16 | 320 | High | ~717K | Semantic part-level fusion (likely best) |
| 4 | H/32 × W/32 | 512 | Very High | ~1.84M | Global object-level integration |

**Expected Results:**
- **Best single stage:** Stage 3 (high-level semantics, good resolution/channel balance)
- **Best multi-stage:** [3,4] or [2,3,4] (avoid stage 1 overhead, maximize semantic fusion)
- **Worst:** Stage 1 alone (too low-level, GAFF overhead not justified)
- **All stages [1,2,3,4]:** May provide best accuracy but high computational cost

### 2.3 Phase 2: Hyperparameter Tuning

**Rationale:**
- SE reduction controls intra-modality attention bottleneck
- Inter-modality sharing affects cross-attention expressiveness
- Merge strategy impacts final feature combination

**Hyperparameter Grid:**

| Parameter | Values | Effect |
|-----------|--------|--------|
| SE_reduction | {4, 8} | Lower = more params, richer intra-attention |
| inter_shared | {False, True} | False = asymmetric cross-attention (more expressive) |
| merge_bottleneck | {False, True} | True = non-linear merge with bottleneck |

**Total Combinations:** 2 × 2 × 2 = 8 configurations per stage config

**Experiment Plan:**
- Top 3 stage configs from Phase 1
- 8 hyperparameter combinations per config
- Minus 1 default (already tested in Phase 1)
- **Total: 3 × 7 = 21 new experiments + 3 defaults = 24 Phase 2 experiments**

**Example: If top 3 configs are s3, s23, s34**

| Exp ID | Stages | SE_r | Inter_Shared | Merge_BN | Code |
|--------|--------|------|--------------|----------|------|
| exp_009 | [3] | 4 | False | True | s3_r4_is0_mb1 |
| exp_010 | [3] | 4 | True | False | s3_r4_is1_mb0 |
| exp_011 | [3] | 4 | True | True | s3_r4_is1_mb1 |
| exp_012 | [3] | 8 | False | False | s3_r8_is0_mb0 |
| exp_013 | [3] | 8 | False | True | s3_r8_is0_mb1 |
| exp_014 | [3] | 8 | True | False | s3_r8_is1_mb0 |
| exp_015 | [3] | 8 | True | True | s3_r8_is1_mb1 |
| ... | ... | ... | ... | ... | (similar for [2,3] and [3,4]) |

**Hypothesis Testing:**

| Hypothesis | Prediction | Metric |
|------------|------------|--------|
| H1: Lower SE reduction improves accuracy | reduction=4 > reduction=8 | mAP |
| H2: Asymmetric cross-attention helps | inter_shared=False > True | mAP |
| H3: Bottleneck merge adds expressiveness | merge_bottleneck=True > False | mAP |
| H4: Higher reduction reduces overfitting | reduction=8 better train-val gap | Loss curves |

---

## 3. Training Configuration

### 3.1 Standard Settings

**Backbone:** mit_b1 (default, configurable)
```python
backbone_name = 'mit_b1'
total_params_base = 69.5M
total_params_with_gaff = 69.5M + GAFF_overhead
```

**Training Hyperparameters:**
```python
epochs = 25 (planned), 15 (actual runs)
batch_size = 8 (planned), 16 (actual runs)
learning_rate = 0.0001 (planned), 0.02 (actual runs)
optimizer = SGD(momentum=0.9, weight_decay=1e-4)
lr_scheduler = StepLR(step_size=15, gamma=0.1)  # Step decay
```

**Data Configuration:**
```python
data_dir = '../RGBX_Semantic_Segmentation/data/images'
labels_dir = '../RGBX_Semantic_Segmentation/data/labels'
train_split = 0.8  # 80% train, 20% test
num_classes = 2  # Background + object class
```

**GAFF Default Parameters:**
```python
# Phase 1: Fixed
se_reduction = 4
inter_shared = False
merge_bottleneck = False

# Phase 2: Variable
se_reduction ∈ {4, 8}
inter_shared ∈ {False, True}
merge_bottleneck ∈ {False, True}
```

### 3.2 Encoder Configuration

**GAFF Integration:**

The encoder checks `gaff_stages` to apply GAFF selectively:

```python
# Example: gaff_stages = [2, 3]

def forward(self, rgb, aux):
    # Stage 1: No fusion (identity)
    rgb_1, aux_1 = stage1(rgb), stage1(aux)

    # Stage 2: GAFF fusion
    if 2 in gaff_stages:
        fused_2 = gaff_stage2(rgb_2, aux_2,
                             se_reduction=se_r,
                             inter_shared=inter_s,
                             merge_bottleneck=merge_b)
        rgb_2, aux_2 = fused_2, fused_2

    # Stage 3: GAFF fusion
    if 3 in gaff_stages:
        fused_3 = gaff_stage3(rgb_3, aux_3, ...)
        rgb_3, aux_3 = fused_3, fused_3

    # Stage 4: No fusion (identity)
    rgb_4, aux_4 = stage4(rgb_3), stage4(aux_3)

    return [fused_1/rgb_1, fused_2/rgb_2, fused_3, fused_4]
```

**Key Decision:** After fusion, both RGB and Aux streams carry identical fused features (symmetric design).

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

**Completed Experiments:** 4/32 (12.5%)

**Phase 1 Progress:**

| Exp ID | Config | Status | Best mAP | Best mAP@50 | Epochs | Notes |
|--------|--------|--------|----------|-------------|--------|-------|
| exp_001 | s1_r4_is0_mb0 | ✅ Complete | TBD | TBD | 15 | Stage 1 only |
| exp_002 | s2_r4_is0_mb0 | ✅ Complete | TBD | TBD | 15 | Stage 2 only |
| exp_003 | s3_r4_is0_mb0 | ✅ Complete | TBD | TBD | 15 | Stage 3 only |
| exp_004 | s4_r4_is0_mb0 | ✅ Complete | TBD | TBD | 15 | Stage 4 only |
| exp_005 | s23_r4_is0_mb0 | ⏸️ Pending | - | - | - | Stages 2+3 |
| exp_006 | s34_r4_is0_mb0 | ⏸️ Pending | - | - | - | Stages 3+4 |
| exp_007 | s234_r4_is0_mb0 | ⏸️ Pending | - | - | - | Stages 2+3+4 |
| exp_008 | s1234_r4_is0_mb0 | ⏸️ Pending | - | - | - | All stages |

**Pending Experiments:** 28/32

**Results Location:** `results/gaff_ablations_full/phase1_stage_selection/exp_XXX_*/`

### 4.2 Results Analysis (Placeholder)

**To be filled after Phase 1 completion:**

**Stage-wise Performance:**
```
Stage Config  | mAP  | mAP@50 | mAP@75 | Params (M) | GAFF Overhead | Time (hrs) | Ranking
--------------|------|--------|--------|------------|---------------|------------|--------
[1]           | ?    | ?      | ?      | ?          | ~29K          | ?          | ?
[2]           | ?    | ?      | ?      | ?          | ~115K         | ?          | ?
[3]           | ?    | ?      | ?      | ?          | ~717K         | ?          | ?
[4]           | ?    | ?      | ?      | ?          | ~1.84M        | ?          | ?
[2,3]         | ?    | ?      | ?      | ?          | ~832K         | ?          | ?
[3,4]         | ?    | ?      | ?      | ?          | ~2.56M        | ?          | ?
[2,3,4]       | ?    | ?      | ?      | ?          | ~2.67M        | ?          | ?
[1,2,3,4]     | ?    | ?      | ?      | ?          | ~2.70M        | ?          | ?
```

**Hyperparameter Sensitivity** (Phase 2):
```
Config | SE_r | Inter_Shared | Merge_BN | mAP  | Δ vs Default | Params | Ranking
-------|------|--------------|----------|------|--------------|--------|--------
Top #1 | 4    | False        | False    | ?    | Baseline     | ?      | ?
Top #1 | 4    | False        | True     | ?    | ?            | ?      | ?
Top #1 | 4    | True         | False    | ?    | ?            | ?      | ?
Top #1 | 4    | True         | True     | ?    | ?            | ?      | ?
Top #1 | 8    | False        | False    | ?    | ?            | ?      | ?
Top #1 | 8    | False        | True     | ?    | ?            | ?      | ?
Top #1 | 8    | True         | False    | ?    | ?            | ?      | ?
Top #1 | 8    | True         | True     | ?    | ?            | ?      | ?
(Similar for Top #2 and Top #3)
```

### 4.3 Visualizations (To be generated)

**1. Stage Performance Comparison:**
- Bar chart: mAP by stage configuration
- Line plot: mAP@{50, 75} by stage
- Scatter: Accuracy vs. parameter overhead

**2. Hyperparameter Sensitivity:**
- Heatmap: mAP for all hyperparameter combinations
- Bar chart: Effect of each hyperparameter in isolation
- Interaction plot: SE_reduction × inter_shared × merge_bottleneck

**3. Training Curves:**
- Loss curves for all 32 experiments
- Validation mAP over epochs
- Overfitting analysis (train-val gap)

**4. Computational Cost:**
- Training time vs. mAP (efficiency analysis)
- Parameters vs. mAP (parameter efficiency)
- Memory usage vs. stage configuration

---

## 5. Implementation Details

### 5.1 Running GAFF Ablations

**Master Script:** `crossattentiondet/ablations/scripts/run_gaff_ablations.py`

**Usage:**
```bash
# Run complete study (Phase 1 + Phase 2)
python crossattentiondet/ablations/scripts/run_gaff_ablations.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-base results/gaff_ablations_full \
    --backbone mit_b1 \
    --epochs 25 \
    --batch-size 8

# Run Phase 1 only
python crossattentiondet/ablations/scripts/run_gaff_ablations.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-base results/gaff_ablations_full \
    --epochs 25
```

**Single Experiment Script:** `crossattentiondet/ablations/scripts/train_gaff_ablation.py`

```bash
# Run specific experiment
python crossattentiondet/ablations/scripts/train_gaff_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/gaff_ablations_full/exp_003 \
    --backbone mit_b1 \
    --epochs 25 \
    --gaff-stages "3" \
    --gaff-se-reduction 4 \
    --gaff-inter-shared false \
    --gaff-merge-bottleneck false
```

### 5.2 Code Locations

**GAFF Module:**
```
crossattentiondet/ablations/fusion/gaff.py
├── SEBlock (lines 22-65)
├── InterModalityAttention (lines 67-121)
├── GAFFBlock (lines 123-220)
└── build_gaff_block (lines 222-249)
```

**Encoder Integration:**
```
crossattentiondet/ablations/encoder_gaff_flexible.py
└── get_gaff_encoder() - Returns encoder with GAFF at specified stages
```

**Ablation Framework:**
```
crossattentiondet/ablations/scripts/run_gaff_ablations.py
├── Phase 1: Stage selection (lines 59-86)
├── Phase 2: Hyperparameter tuning (lines 88-121)
└── Experiment management (logging, checkpointing, summary)
```

**Training Script:**
```
crossattentiondet/ablations/scripts/train_gaff_ablation.py
├── Logger class (lines 48-157)
├── GAFFAblationTrainer (lines 159-end)
└── Comprehensive logging to CSV, JSON
```

### 5.3 Output Structure

**Results Directory:**
```
results/gaff_ablations_full/
├── phase1_stage_selection/
│   ├── exp_001_s1_r4_is0_mb0/
│   │   ├── checkpoints/
│   │   │   └── best_model.pth
│   │   ├── training.log
│   │   ├── config.json
│   │   ├── model_info.json
│   │   ├── metrics_per_epoch.csv
│   │   ├── metrics_per_batch.csv
│   │   └── final_results.json
│   ├── exp_002_s2_r4_is0_mb0/
│   │   └── ...
│   └── ...
├── phase2_hyperparameter_tuning/
│   ├── exp_009_s3_r4_is0_mb1/
│   │   └── ...
│   └── ...
├── master_log.txt
└── summary_all_experiments.csv
```

**Key Files:**
- `master_log.txt`: Overall progress of all experiments
- `summary_all_experiments.csv`: Aggregated results for all experiments
- `config.json`: Experiment configuration
- `model_info.json`: Model architecture details and parameter counts
- `metrics_per_epoch.csv`: Training metrics per epoch
- `final_results.json`: Final COCO evaluation results

---

## 6. Analysis Guidelines

### 6.1 Phase 1 Analysis

**After all 8 experiments complete:**

1. **Rank by mAP:**
   - Identify top 3 stage configurations
   - Check if hypothesis (stage 3 or [3,4] best) holds

2. **Analyze by object size:**
   - Do early stages (1,2) help small objects?
   - Do late stages (3,4) help large objects?

3. **Computational cost:**
   - Is all-stage fusion [1,2,3,4] worth the 2.7M parameter overhead?
   - Best accuracy/cost trade-off?

4. **Training stability:**
   - Which configs converge fastest?
   - Any overfitting in multi-stage configs?

5. **Select top 3 for Phase 2:**
   - Highest mAP
   - Diversity in stage patterns if close (e.g., single-stage, two-stage, multi-stage)

### 6.2 Phase 2 Analysis

**After hyperparameter experiments:**

1. **SE Reduction Ratio:**
   - Does reduction=4 (richer) outperform reduction=8 (compressed)?
   - Trade-off: accuracy gain vs. parameter cost

2. **Inter-modality Sharing:**
   - Does separate (asymmetric) beat shared (symmetric)?
   - Interpretation: Are RGB and Aux equally important, or does one dominate?

3. **Merge Strategy:**
   - Does bottleneck merge improve over direct merge?
   - Is the non-linearity worth the extra parameters?

4. **Hyperparameter Interactions:**
   - Are there synergistic combinations?
   - Example: Does reduction=8 + bottleneck=True compensate for compression?

5. **Final recommendation:**
   - Best overall configuration (stage + hyperparameters)
   - Expected mAP improvement vs. baseline
   - Recommended config for production use

### 6.3 Comparison with Baseline and CSSA

**Baseline (no fusion):**
- Use mit_b1 trained without any fusion
- Compare mAP, training time, parameters

**GAFF vs. Baseline:**
```
Metric          | Baseline | Best GAFF | Improvement | Cost
----------------|----------|-----------|-------------|-----
mAP             | ?        | ?         | ?           | +0.7-2.7M params
mAP@50          | ?        | ?         | ?           | +5-10% train time
mAP@75          | ?        | ?         | ?           |
Params (M)      | 69.5     | 70.2-72.2 | +1-4%       |
Time (hrs)      | ?        | ?         | +10-15%     |
```

**GAFF vs. CSSA:**
```
Metric          | CSSA     | GAFF      | Winner      | Trade-off
----------------|----------|-----------|-------------|----------
mAP             | ?        | ?         | ?           | Accuracy
mAP@50          | ?        | ?         | ?           |
Params Added    | ~2K      | ~0.7-2.7M | CSSA        | Efficiency
Time (hrs)      | ?        | ?         | ?           | Speed
Complexity      | Low      | Medium    | CSSA        | Simplicity
```

**Questions to Answer:**
- Is GAFF's higher complexity justified by accuracy gains?
- When to use CSSA (lightweight) vs. GAFF (accurate)?
- Does GAFF scale better to larger backbones?

---

## 7. Troubleshooting & FAQs

### 7.1 Common Issues

**Q: Experiment fails with "dimension mismatch"**
A: Check that:
- `gaff_stages` contains valid stage numbers [1,2,3,4]
- Channel dimensions match between encoder stages and GAFF modules
- `out_channels` is set correctly (usually same as `in_channels`)

**Q: Training is very slow**
A: GAFF is heavier than CSSA. To speed up:
- Use reduction=8 instead of 4 (fewer SE params)
- Enable `inter_shared=True` (same param count, but may be faster)
- Reduce number of fusion stages (e.g., [3] instead of [1,2,3,4])
- Increase batch size if memory allows

**Q: CUDA Out of Memory**
A: GAFF adds significant memory overhead:
- Reduce batch size (e.g., 8 → 4 → 2)
- Use gradient accumulation to maintain effective batch size
- Apply fusion only at later stages (3 or 4) to reduce overhead
- Consider using a smaller backbone (mit_b0 instead of mit_b1)

**Q: mAP is lower than baseline**
A: Possible causes:
- GAFF may require more training epochs to converge (try 30-40 epochs)
- Learning rate may need tuning (try 0.005 or 0.001)
- Hyperparameters may not be optimal (run Phase 2)
- Dataset may not benefit from multi-modal fusion

**Q: All stage configurations give similar results**
A: This suggests:
- Multi-modal fusion has limited benefit for this dataset
- GAFF may be overkill; try simpler CSSA instead
- Auxiliary modalities (thermal, event) may not be complementary enough

### 7.2 Debug Mode

**Enable detailed logging:**
```bash
python crossattentiondet/ablations/scripts/train_gaff_ablation.py \
    --data ... \
    --output-dir results/debug \
    --gaff-stages "3" \
    # Logging is already comprehensive by default
```

**Visualize attention maps:**
```python
# In GAFFBlock, save intermediate outputs
def forward(self, x_rgb, x_aux):
    x_rgb_se = self.se_rgb(x_rgb)
    x_aux_se = self.se_aux(x_aux)

    # Save for visualization
    torch.save(x_rgb_se, 'debug/x_rgb_se.pt')
    torch.save(x_aux_se, 'debug/x_aux_se.pt')

    w_rgb_from_aux, w_aux_from_rgb = self.inter_attn(x_rgb, x_aux)

    torch.save(w_rgb_from_aux, 'debug/w_rgb_from_aux.pt')
    torch.save(w_aux_from_rgb, 'debug/w_aux_from_rgb.pt')

    # ... rest of forward pass
```

**Analyze attention statistics:**
```python
import torch
import matplotlib.pyplot as plt

# Load attention weights
w = torch.load('debug/w_rgb_from_aux.pt')  # (B, C, H, W)

# Spatial average
w_avg = w.mean(dim=[2, 3])  # (B, C)

# Plot channel-wise attention
plt.figure(figsize=(12, 4))
plt.bar(range(w_avg.shape[1]), w_avg[0].cpu().numpy())
plt.xlabel('Channel')
plt.ylabel('Avg Attention Weight')
plt.title('RGB←Aux Cross-Attention Weights')
plt.show()
```

### 7.3 Extending GAFF

**Add new hyperparameters:**

1. **Variable output channels:**
   ```python
   # In run_gaff_ablations.py
   gaff_block = GAFFBlock(in_channels=320, out_channels=256)
   # Compress or expand channels during fusion
   ```

2. **Different SE reduction per stage:**
   ```python
   # Stage-specific SE reduction
   stage_se_reductions = {1: 2, 2: 4, 3: 8, 4: 16}
   # More params in early stages, fewer in late stages (or vice versa)
   ```

3. **Learnable merge weights:**
   ```python
   # Instead of simple addition in guided fusion
   self.alpha_rgb = nn.Parameter(torch.ones(1))
   self.alpha_aux = nn.Parameter(torch.ones(1))

   x_rgb_guided = x_rgb_se + self.alpha_rgb * w_rgb_from_aux * x_aux
   x_aux_guided = x_aux_se + self.alpha_aux * w_aux_from_rgb * x_rgb
   ```

---

## 8. Future Work & Extensions

### 8.1 Potential Improvements

1. **Lightweight GAFF:**
   - Remove SE blocks, keep only inter-modality attention
   - Reduce to "GAFF-Lite" for better speed/accuracy trade-off

2. **Asymmetric guided fusion:**
   - Use different weights for RGB and Aux guided fusion
   - `x_rgb_guided = α * x_rgb_se + β * w_rgb_from_aux * x_aux`

3. **Multi-scale inter-modality attention:**
   - Apply cross-attention at multiple spatial resolutions
   - Pyramid-style attention for better scale invariance

4. **Gated fusion:**
   - Add learnable gates to decide fusion strength
   - `gate = sigmoid(conv(concat[x_rgb, x_aux]))`
   - `fused = gate * gaff_fusion + (1-gate) * x_rgb`

### 8.2 Comparison Studies

**GAFF vs. Other Fusion Methods:**
1. Simple concatenation + 1×1 conv
2. Element-wise addition
3. CSSA (see CSSA_ABLATION_GUIDE.md)
4. Full cross-attention (transformer-style)
5. Late concatenation (FPN-level fusion)

**Create comparison table:**
```
Method          | mAP  | Params | Time | Complexity | Use Case
----------------|------|--------|------|------------|----------
Concat+Conv     | ?    | +2C²   | ?    | Low        | Baseline
Element-wise Add| ?    | 0      | ?    | Very Low   | Minimal fusion
CSSA            | ?    | +2Ck   | ?    | Low        | Lightweight
GAFF            | ?    | +7C²   | ?    | Medium     | Accuracy-focused
Transformer     | ?    | +HIGH  | ?    | High       | Maximum capacity
```

### 8.3 Application to Other Backbones

**Test GAFF on different backbones:**
```bash
# Lightweight (fewer channels, lower GAFF overhead)
python run_gaff_ablations.py --backbone mit_b0 --epochs 25

# Deeper (more channels, higher GAFF overhead, may need memory optimization)
python run_gaff_ablations.py --backbone mit_b2 --epochs 25 --batch-size 4
```

**Hypothesis:** GAFF benefits may scale with backbone capacity, as richer features allow better cross-modal interaction.

### 8.4 Task-Specific Adaptations

**Semantic Segmentation:**
- Apply GAFF at decoder stages as well
- Multi-scale fusion across encoder-decoder connections

**Instance Segmentation:**
- Add GAFF to RoI features
- Object-level cross-modal attention

**Panoptic Segmentation:**
- Separate GAFF for stuff vs. things classes
- Class-conditional fusion weights

---

## Appendix: Quick Reference

### Experiment Naming Convention
```
exp_<XXX>_s<stage_label>_r<se_reduction>_is<inter_shared>_mb<merge_bottleneck>

Examples:
- exp_001_s1_r4_is0_mb0     → Stage 1, SE_r=4, not shared, no bottleneck
- exp_005_s23_r4_is0_mb0    → Stages 2+3, default hyperparameters
- exp_009_s3_r8_is1_mb1     → Stage 3, SE_r=8, shared, bottleneck

Stage labels:
s1    = [1]
s2    = [2]
s3    = [3]
s4    = [4]
s23   = [2, 3]
s34   = [3, 4]
s234  = [2, 3, 4]
s1234 = [1, 2, 3, 4]

Hyperparameter codes:
r4  = SE reduction ratio 4
r8  = SE reduction ratio 8
is0 = inter_shared False (separate convs)
is1 = inter_shared True (shared conv)
mb0 = merge_bottleneck False (direct merge)
mb1 = merge_bottleneck True (bottleneck merge)
```

### Commands Cheat Sheet
```bash
# Run complete ablation study
python crossattentiondet/ablations/scripts/run_gaff_ablations.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-base results/gaff_ablations_full \
    --epochs 25

# Run single experiment
python crossattentiondet/ablations/scripts/train_gaff_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/test_exp \
    --gaff-stages "3" \
    --gaff-se-reduction 4 \
    --gaff-inter-shared false \
    --gaff-merge-bottleneck false \
    --epochs 25

# Check results
cat results/gaff_ablations_full/summary_all_experiments.csv

# Monitor training
tail -f results/gaff_ablations_full/phase1_stage_selection/exp_001_*/training.log

# View final results
cat results/gaff_ablations_full/phase1_stage_selection/exp_001_*/final_results.json
```

### Key Files
```
crossattentiondet/ablations/fusion/gaff.py                    # GAFF module
crossattentiondet/ablations/encoder_gaff_flexible.py          # Encoder with GAFF
crossattentiondet/ablations/scripts/run_gaff_ablations.py     # Master ablation runner
crossattentiondet/ablations/scripts/train_gaff_ablation.py    # Single experiment trainer
results/gaff_ablations_full/                                  # Results directory
```

### Parameter Estimation Formula

For a GAFF block at stage with C channels:

```python
# SE blocks (2 total: RGB + Aux)
se_params = 2 * (2 * C^2 / reduction)

# Inter-modality attention
inter_params = 4 * C^2  # Same for shared or separate

# Merge layer
if merge_bottleneck:
    merge_params = 2*C^2 + C*out_C + 2*C + 2*out_C
else:
    merge_params = 2*C*out_C + 2*out_C

# Total (assuming out_C = C)
total = 2*(2*C^2/reduction) + 4*C^2 + merge_params
```

**Example (mit_b1 stage 3: C=320, reduction=4, out_C=320):**
- SE: 2 × (2×320²/4) = 102,400
- Inter: 4×320² = 409,600
- Merge (direct): 2×320×320 + 2×320 = 205,440
- **Total: 717,440 parameters**

---

**Document End**
