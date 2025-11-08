# GAFF Ablation Study: Comprehensive Guide

Complete documentation for the GAFF (Guided Attentive Feature Fusion) ablation study implementation.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [CPU Testing Protocol](#cpu-testing-protocol)
4. [Experiment Design](#experiment-design)
5. [Implementation Details](#implementation-details)
6. [GPU Deployment](#gpu-deployment)
7. [Results Analysis](#results-analysis)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What is GAFF?

GAFF (Guided Attentive Feature Fusion) is a feature fusion method for multispectral object detection introduced in WACV 2021. It learns to dynamically weight and combine features from different modalities (e.g., RGB and thermal) using attention mechanisms.

**Key Innovation**: Unlike fixed fusion strategies, GAFF uses learned attention to adapt fusion based on context and imaging conditions.

### Paper Details

- **Title**: "Guided Attentive Feature Fusion for Multispectral Pedestrian Detection"
- **Authors**: Heng Zhang, Elisa Fromont, Sébastien Lefèvre, Bruno Avignon
- **Conference**: WACV 2021
- **GitHub**: https://github.com/zhanghengdev/GAFF
- **Performance**: Ranked #3 on KAIST benchmark at time of publication

---

## Architecture

### GAFF Block Components

#### 1. Intra-Modality Attention (SE Block)

SE (Squeeze-and-Excitation) blocks apply channel attention within each modality:

```
Input (B, C, H, W)
  ↓
Global Average Pooling → (B, C)
  ↓
FC (C → C/r) → ReLU → FC (C/r → C) → Sigmoid
  ↓
Channel weights (B, C, 1, 1)
  ↓
Element-wise multiplication with input
  ↓
Attention-weighted output (B, C, H, W)
```

**Hyperparameter**: `se_reduction` (r) - typically 4 or 8

#### 2. Inter-Modality Attention

Cross-attention between modalities:

```
x_rgb (B, C, H, W)  +  x_aux (B, C, H, W)
          ↓
    Concatenate → (B, 2C, H, W)
          ↓
    1×1 Conv → Sigmoid
          ↓
    w_rgb←aux (B, C, H, W)  +  w_aux←rgb (B, C, H, W)
```

**Hyperparameter**: `inter_shared` - use shared or separate convolutions

#### 3. Guided Fusion

Combine intra- and inter-modality information:

```
x̂_rgb = SE_rgb(x_rgb) + w_rgb←aux * x_aux
x̂_aux = SE_aux(x_aux) + w_aux←rgb * x_rgb
```

#### 4. Merge

Final fusion step:

```
Concat[x̂_rgb, x̂_aux] → (B, 2C, H, W)
           ↓
      1×1 Conv → BN
           ↓
    Output (B, C, H, W)
```

**Hyperparameter**: `merge_bottleneck` - direct or bottleneck pathway

### Full GAFF Block Diagram

```
RGB (B,C,H,W)          Aux (B,C,H,W)
    ↓                      ↓
  SE_rgb                SE_aux     (Intra-modality)
    ↓                      ↓
    └──────────┬───────────┘
               ↓
        Inter-Modality         (Inter-modality)
         Attention
               ↓
    ┌──────────┴───────────┐
    ↓                      ↓
x̂_rgb                   x̂_aux     (Guided fusion)
    └──────────┬───────────┘
               ↓
           Concat
               ↓
           1×1 Conv
               ↓
         Fused Output
```

---

## CPU Testing Protocol

### Why CPU Testing?

GPU resources are expensive. Thorough CPU testing ensures code is error-free before deployment, saving time and money.

### Testing Levels

#### Level 1: Module Tests (5-10 mins)

Test individual GAFF fusion blocks:

```bash
# Unit tests - basic functionality
python -m crossattentiondet.ablations.fusion.test_gaff

# Detailed verification - all configurations
python -m crossattentiondet.ablations.fusion.verify_gaff
```

**What's tested**:
- Shape correctness
- Gradient flow
- Parameter counts
- All hyperparameter combinations
- CPU performance benchmarks

**Success criteria**: All tests pass, no NaN/Inf values

#### Level 2: Encoder Tests (10-15 mins)

Test GAFF integration into full encoder:

```bash
python -m crossattentiondet.ablations.scripts.verify_gaff_encoder
```

**What's tested**:
- All backbones (mit_b0-b5)
- All stage configurations
- Forward/backward passes
- Output shape verification
- Parameter analysis

**Success criteria**: All backbones and configurations work correctly

#### Level 3: Training Loop Tests (15-20 mins)

Test complete training pipeline with synthetic data:

```bash
python -m crossattentiondet.ablations.scripts.dry_run_gaff
```

**What's tested**:
- Full forward-backward-optimize loop
- Synthetic dataset generation
- Loss computation
- Gradient updates
- Checkpoint save/load

**Success criteria**: Training loop completes without errors

#### Level 4: Complete Checklist (30-40 mins)

Run all tests automatically:

```bash
python -m crossattentiondet.ablations.scripts.pre_gpu_checklist
```

**What's tested**: Everything from Levels 1-3

**Success criteria**: All checks pass ✓

### Expected Timeline

| Stage | Duration | Cumulative |
|-------|----------|------------|
| Module tests | 5-10 min | 5-10 min |
| Encoder tests | 10-15 min | 15-25 min |
| Dry-run tests | 15-20 min | 30-45 min |
| **Total** | **30-45 min** | **30-45 min** |

---

## Experiment Design

### Ablation Matrix

#### Phase 1: Stage Selection (8 experiments)

Test which stages benefit from GAFF fusion:

| Exp # | Stages | Description |
|-------|--------|-------------|
| 001 | [1] | GAFF at stage 1 only |
| 002 | [2] | GAFF at stage 2 only |
| 003 | [3] | GAFF at stage 3 only |
| 004 | [4] | GAFF at stage 4 only |
| 005 | [2,3] | GAFF at stages 2&3 |
| 006 | [3,4] | GAFF at stages 3&4 |
| 007 | [2,3,4] | GAFF at stages 2,3,4 |
| 008 | [1,2,3,4] | GAFF at all stages |

**Config**: All use default `se_reduction=4`, `inter_shared=False`, `merge_bottleneck=False`

#### Phase 2: Hyperparameter Tuning (24 experiments)

For top 3 stage configs from Phase 1, test hyperparameter variants:

| SE Reduction | Inter Shared | Merge Bottleneck | # Configs |
|--------------|--------------|------------------|-----------|
| 4, 8 | False, True | False, True | 2×2×2 = 8 |

**Total**: 3 stages × 8 configs = 24 experiments

#### Total Experiments

- Phase 1: 8 experiments
- Phase 2: 24 experiments
- **Grand Total**: 32 experiments

### Experiment Naming

Format: `gaff_exp_{num:03d}_s{stages}_r{reduction}_is{shared}_mb{bottleneck}`

Examples:
- `gaff_exp_001_s1_r4_is0_mb0` - Stage 1, SE reduction 4, no sharing, no bottleneck
- `gaff_exp_008_s1234_r4_is0_mb0` - All stages, defaults
- `gaff_exp_012_s34_r8_is1_mb1` - Stages 3&4, reduction 8, shared, bottleneck

### Training Configuration

**Standard settings**:
- Backbone: `mit_b1`
- Epochs: 25
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: AdamW
- Scheduler: Polynomial LR decay

---

## Implementation Details

### Module Files

#### 1. `fusion/gaff.py`

Core GAFF implementation (~200 lines):

```python
# Main components
class SEBlock(nn.Module):
    """Intra-modality channel attention"""

class InterModalityAttention(nn.Module):
    """Cross-attention between modalities"""

class GAFFBlock(FusionBlock):
    """Complete GAFF fusion block"""
```

**Key methods**:
- `forward(x_rgb, x_aux)` → fused features
- Inherits from `FusionBlock` base class

#### 2. `encoder_gaff_flexible.py`

Flexible encoder with configurable GAFF stages (~350 lines):

```python
class RGBXTransformerGAFFFlexible(nn.Module):
    """Encoder with stage-wise GAFF selection"""

    def __init__(
        self,
        gaff_stages=[4],  # Which stages use GAFF
        gaff_se_reduction=4,
        gaff_inter_shared=False,
        gaff_merge_bottleneck=False,
        ...
    ):
```

**Variants**: `mit_b0_gaff_flexible` through `mit_b5_gaff_flexible`

### Testing Files

#### 1. `fusion/test_gaff.py`

Unit tests (~150 lines):
- 30+ test cases
- Module-level verification
- CPU stress tests

#### 2. `fusion/verify_gaff.py`

Module verification (~200 lines):
- Hyperparameter sweep
- Performance benchmarks
- Memory profiling

#### 3. `scripts/verify_gaff_encoder.py`

Encoder verification (~350 lines):
- All backbone tests
- Stage configuration tests
- Gradient flow verification

#### 4. `scripts/dry_run_gaff.py`

Training pipeline tests (~400 lines):
- Synthetic data generation
- Complete training loop
- Checkpoint operations

#### 5. `scripts/pre_gpu_checklist.py`

Master verification script (~150 lines):
- Runs all test suites
- Generates pass/fail report
- Ready-for-GPU indicator

---

## GPU Deployment

### Prerequisites

✓ All CPU tests passed
✓ Dataset prepared (KAIST, MFNet, etc.)
✓ GPU environment set up

### Single Experiment Training

```bash
python -m crossattentiondet.ablations.scripts.train_gaff_ablation \
    --data-root /path/to/dataset \
    --backbone mit_b1 \
    --gaff-stages 4 \
    --gaff-se-reduction 4 \
    --gaff-inter-shared false \
    --gaff-merge-bottleneck false \
    --epochs 25 \
    --batch-size 8 \
    --lr 0.0001 \
    --output-dir ./results/gaff_exp_001
```

### Full Ablation Suite

```bash
python -m crossattentiondet.ablations.scripts.run_gaff_ablations \
    --data-root /path/to/dataset \
    --backbone mit_b1 \
    --epochs 25 \
    --batch-size 8 \
    --output-dir ./results/gaff_ablations
```

This will:
1. Run Phase 1 (8 stage experiments)
2. Identify top 3 stage configurations
3. Run Phase 2 (24 hyperparameter experiments)
4. Generate summary CSV and plots

### Expected Runtime

**Per experiment** (mit_b1, 25 epochs, KAIST dataset):
- 1× A100 GPU: ~2-3 hours
- 1× V100 GPU: ~4-6 hours
- 1× RTX 3090: ~3-5 hours

**Full ablation** (32 experiments):
- 1× A100 GPU: ~64-96 hours (3-4 days)
- 4× A100 GPUs (parallel): ~16-24 hours (1 day)

### Output Structure

```
results/gaff_ablations/
├── phase1_stage_selection/
│   ├── exp_001_s1/
│   │   ├── config.json
│   │   ├── training.log
│   │   ├── metrics_per_epoch.csv
│   │   ├── final_results.json
│   │   └── checkpoint_best.pth
│   ├── exp_002_s2/
│   └── ...
│
├── phase2_hyperparameter_tuning/
│   ├── exp_009_s4_r8_is0_mb0/
│   └── ...
│
└── summary/
    ├── all_results.csv
    ├── stage_comparison.png
    ├── hyperparameter_heatmap.png
    └── final_report.md
```

---

## Results Analysis

### Metrics Tracked

**Per-epoch metrics**:
- Training loss
- Validation mAP (@ IoU 0.5, 0.75, 0.5:0.95)
- Precision, Recall
- Inference time

**Per-experiment outputs**:
- Best validation mAP
- Final model checkpoint
- Training curves
- Model parameter count
- FLOPs

### Analysis Scripts

```bash
# Generate comparison tables
python -m crossattentiondet.ablations.scripts.analyze_gaff_results \
    --results-dir ./results/gaff_ablations

# Create publication-ready plots
python -m crossattentiondet.ablations.scripts.plot_gaff_results \
    --results-dir ./results/gaff_ablations \
    --output-dir ./figures
```

### Expected Findings

Based on the GAFF paper, we expect:

1. **Stage selection**: Later stages (3&4) likely benefit most from GAFF
2. **SE reduction**: Lower reduction (4) may work better than higher (8)
3. **Inter-modality sharing**: Separate convs likely outperform shared
4. **Merge pathway**: Direct merge likely sufficient (no bottleneck needed)

---

## Troubleshooting

### Common Issues

#### 1. Import errors

**Error**: `ModuleNotFoundError: No module named 'crossattentiondet'`

**Solution**:
```bash
# Add to PYTHONPATH
export PYTHONPATH=/path/to/cmx-object-detection:$PYTHONPATH

# Or install as package
pip install -e .
```

#### 2. CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Use smaller backbone: `--backbone mit_b0`
- Use gradient accumulation
- Enable mixed precision training

#### 3. Slow CPU tests

**Issue**: Tests taking > 1 hour on CPU

**Solutions**:
- This is normal for CPU! Wait it out or...
- Reduce test iterations in verification scripts
- Skip some test configurations
- Use faster CPU or small instance

#### 4. NaN loss during training

**Error**: Loss becomes NaN

**Solutions**:
- Reduce learning rate: `--lr 0.00001`
- Add gradient clipping
- Check data normalization
- Verify no corrupted images in dataset

#### 5. Different results from paper

**Issue**: mAP doesn't match paper

**Possible causes**:
- Different dataset split
- Different preprocessing
- Different training duration
- Different backbone
- Implementation differences (normal!)

---

## Appendix

### Hardware Requirements

**Minimum (CPU testing only)**:
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 10GB

**Recommended (GPU training)**:
- GPU: 12GB+ VRAM (RTX 3090, V100, A100)
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 100GB+ (for dataset + results)

### Software Dependencies

```
torch >= 1.8.0
torchvision >= 0.9.0
numpy
opencv-python
pillow
matplotlib
tqdm
```

### References

1. **GAFF Paper**: Zhang et al., "Guided Attentive Feature Fusion for Multispectral Pedestrian Detection", WACV 2021
2. **SE-Net**: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
3. **SegFormer**: Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021

### Contact

For questions or issues:
- Check test outputs for error messages
- Review this guide for solutions
- Compare with CSSA implementation (`ablations/fusion/cssa.py`)
- Refer to original GAFF paper and code

---

**Last Updated**: 2025-11-07
**Version**: 1.0
