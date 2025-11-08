# Experimental Matrix: CrossAttentionDet Training & Ablation Studies

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Total Experiments:** 48 configured (9 complete, 39 remaining)

---

## Overview

This document provides a comprehensive view of all experimental configurations in the CrossAttentionDet multi-modal object detection project. The experimental design follows a systematic approach to explore:

1. **Backbone Architecture Variations** (5 variants)
2. **CSSA Fusion Ablations** (11 experiments)
3. **GAFF Fusion Ablations** (32 experiments)

**Experimental Scope:**
- **Theoretical Maximum:** 440 experiments (5 backbones × 2 fusion types × 44 configurations)
- **Practical Implementation:** 48 experiments (smart reduction via two-phase approach)
- **Completion Rate:** 18.75% (9/48 experiments)

---

## 1. Baseline Backbone Training Matrix

### Experiment Configuration
| Experiment | Backbone | Params (M) | Status | Epochs | Best Loss | Time (hrs) | Notes |
|------------|----------|------------|--------|--------|-----------|------------|-------|
| baseline_b0 | mit_b0 | 55.7 | ✅ Complete | 15/15 | 0.1057 | 2.68 | Success |
| baseline_b1 | mit_b1 | 69.5 | ✅ Complete | 15/15 | 0.1027 | 3.44 | Success |
| baseline_b2 | mit_b2 | 82.1 | 🔄 Training | 1/15 | 0.3787 | - | Restarted w/ grad accum |
| baseline_b4 | mit_b4 | 155.4 | ❌ Failed | 0/15 | - | - | CUDA OOM (79.25 GiB exhausted) |
| baseline_b5 | mit_b5 | 196.6 | ❌ Failed | 0/15 | - | - | CUDA OOM (79.25 GiB exhausted) |

**Training Configuration (Original):**
- Batch size: 16
- Learning rate: 0.02
- Epochs: 15
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- LR scheduler: StepLR (step_size=15, gamma=0.1)

**Training Configuration (mit_b2 Restart - Memory Optimized):**
- Batch size: 4 (reduced from 16)
- Gradient accumulation: 4 steps (effective batch size = 16)
- Learning rate: 0.02
- Epochs: 15
- Gradient checkpointing: Available (not enabled in current run)

**Status Summary:**
- ✅ Success: 2/5 (40%)
- 🔄 In Progress: 1/5 (20%)
- ❌ Failed: 2/5 (40%)

---

## 2. CSSA Ablation Matrix

**Total Experiments:** 11
**Phase 1 (Stage Selection):** 7 experiments
**Phase 2 (Threshold Tuning):** 4 experiments
**Completed:** 3/11 (27.3%)

### Phase 1: Stage Selection (Threshold = 0.5, Kernel = 3)

| Exp ID | Code | Stages | Threshold | Kernel | Status | Best mAP | Time (hrs) | Notes |
|--------|------|--------|-----------|--------|--------|----------|------------|-------|
| exp_001 | s1_t0.5 | [1] | 0.5 | 3 | ✅ Complete | - | - | Stage 1 only |
| exp_002 | s2_t0.5 | [2] | 0.5 | 3 | ✅ Complete | - | - | Stage 2 only |
| exp_003 | s3_t0.5 | [3] | 0.5 | 3 | ✅ Complete | - | - | Stage 3 only |
| exp_004 | s4_t0.5 | [4] | 0.5 | 3 | ⏸️ Pending | - | - | Stage 4 only |
| exp_005 | s23_t0.5 | [2,3] | 0.5 | 3 | ⏸️ Pending | - | - | Stages 2+3 |
| exp_006 | s34_t0.5 | [3,4] | 0.5 | 3 | ⏸️ Pending | - | - | Stages 3+4 |
| exp_007 | s1234_t0.5 | [1,2,3,4] | 0.5 | 3 | ⏸️ Pending | - | - | All stages |

### Phase 2: Threshold Sensitivity (Top 2 Configurations)

**Note:** Phase 2 experiments will test threshold values {0.3, 0.7} on the top 2 performing stage configurations from Phase 1.

| Exp ID | Code | Stages | Threshold | Kernel | Status | Best mAP | Notes |
|--------|------|--------|-----------|--------|--------|----------|-------|
| exp_008 | TBD | TBD | 0.3 | 3 | ⏸️ Pending | - | Top config, low threshold |
| exp_009 | TBD | TBD | 0.7 | 3 | ⏸️ Pending | - | Top config, high threshold |
| exp_010 | TBD | TBD | 0.3 | 3 | ⏸️ Pending | - | 2nd best config, low threshold |
| exp_011 | TBD | TBD | 0.7 | 3 | ⏸️ Pending | - | 2nd best config, high threshold |

**Training Configuration:**
- Backbone: mit_b1 (69.5M parameters)
- Batch size: 2
- Learning rate: 0.005
- Epochs: 25
- Optimizer: SGD (momentum=0.9, weight_decay=1e-4)

**CSSA Parameters Explained:**
- **Stages:** Which encoder stages use CSSA fusion vs. identity passthrough
- **Threshold:** Channel switching threshold (higher = more conservative switching)
- **Kernel:** ECA 1D convolution kernel size (must be odd)

**Results Location:** `results/cssa_ablations/exp_XXX_*/`

---

## 3. GAFF Ablation Matrix

**Total Experiments:** 32
**Phase 1 (Stage Selection):** 8 experiments
**Phase 2 (Hyperparameter Tuning):** 24 experiments
**Completed:** 4/32 (12.5%)

### Phase 1: Stage Selection (Default Hyperparameters)

**Default Hyperparameters:** SE_reduction=4, inter_shared=False, merge_bottleneck=False

| Exp ID | Code | Stages | SE_r | Inter_Shared | Merge_BN | Status | Best mAP | Time (hrs) | Notes |
|--------|------|--------|------|--------------|----------|--------|----------|------------|-------|
| exp_001 | s1_r4_is0_mb0 | [1] | 4 | False | False | ✅ Complete | - | - | Stage 1 only |
| exp_002 | s2_r4_is0_mb0 | [2] | 4 | False | False | ✅ Complete | - | - | Stage 2 only |
| exp_003 | s3_r4_is0_mb0 | [3] | 4 | False | False | ✅ Complete | - | - | Stage 3 only |
| exp_004 | s4_r4_is0_mb0 | [4] | 4 | False | False | ✅ Complete | - | - | Stage 4 only |
| exp_005 | s23_r4_is0_mb0 | [2,3] | 4 | False | False | ⏸️ Pending | - | - | Stages 2+3 |
| exp_006 | s34_r4_is0_mb0 | [3,4] | 4 | False | False | ⏸️ Pending | - | - | Stages 3+4 |
| exp_007 | s234_r4_is0_mb0 | [2,3,4] | 4 | False | False | ⏸️ Pending | - | - | Stages 2+3+4 |
| exp_008 | s1234_r4_is0_mb0 | [1,2,3,4] | 4 | False | False | ⏸️ Pending | - | - | All stages |

### Phase 2: Hyperparameter Tuning (Top 3 Stage Configurations)

**Note:** Phase 2 experiments test all 8 hyperparameter combinations (2³ = 8) on the top 3 stage configurations from Phase 1. Each configuration minus the default (already tested) = 7 new experiments per config × 3 configs = 21 experiments. Plus the 3 defaults already tested in Phase 1.

**Hyperparameter Combinations:**
- SE Reduction: {4, 8}
- Inter-Modality Shared: {False, True}
- Merge Bottleneck: {False, True}
- Total combinations: 2 × 2 × 2 = 8

| Exp ID | Stages | SE_r | Inter_Shared | Merge_BN | Status | Notes |
|--------|--------|------|--------------|----------|--------|-------|
| exp_009-011 | Top Config #1 | Varies | Varies | Varies | ⏸️ Pending | 7 variants (excl. default) |
| exp_012-018 | Top Config #2 | Varies | Varies | Varies | ⏸️ Pending | 7 variants (excl. default) |
| exp_019-025 | Top Config #3 | Varies | Varies | Varies | ⏸️ Pending | 7 variants (excl. default) |

**Example Hyperparameter Grid (for one stage config):**

| SE_r | Inter_Shared | Merge_BN | Exp Code Example |
|------|--------------|----------|------------------|
| 4 | False | False | r4_is0_mb0 (default) |
| 4 | False | True | r4_is0_mb1 |
| 4 | True | False | r4_is1_mb0 |
| 4 | True | True | r4_is1_mb1 |
| 8 | False | False | r8_is0_mb0 |
| 8 | False | True | r8_is0_mb1 |
| 8 | True | False | r8_is1_mb0 |
| 8 | True | True | r8_is1_mb1 |

**Training Configuration:**
- Backbone: mit_b1 (69.5M parameters, configurable)
- Batch size: 8 (default), 16 (actual runs)
- Learning rate: 0.0001 (default), 0.02 (actual runs)
- Epochs: 25 (default), 15 (actual runs)
- Optimizer: SGD (momentum=0.9, weight_decay=1e-4)

**GAFF Parameters Explained:**
- **Stages:** Which encoder stages use GAFF fusion vs. identity passthrough
- **SE_r:** Squeeze-Excitation reduction ratio (higher = more compression, fewer params)
- **Inter_Shared:** Share conv weights for RGB→Aux and Aux→RGB cross-attention (True = fewer params)
- **Merge_BN:** Use bottleneck merge pathway (True = 2C→C→out_channels vs False = 2C→out_channels)

**Results Location:** `results/gaff_ablations_full/exp_XXX_*/`

---

## 4. Complete Experimental Space Visualization

### 4.1 Current Implementation (48 Experiments)

```
CrossAttentionDet Experiments (48 total)
│
├── Baseline Backbones (5)
│   ├── ✅ mit_b0 (55.7M params)
│   ├── ✅ mit_b1 (69.5M params)
│   ├── 🔄 mit_b2 (82.1M params)
│   ├── ❌ mit_b4 (155.4M params) - OOM
│   └── ❌ mit_b5 (196.6M params) - OOM
│
├── CSSA Ablations (11)
│   ├── Phase 1: Stage Selection (7)
│   │   ├── ✅ exp_001: Stage 1 only
│   │   ├── ✅ exp_002: Stage 2 only
│   │   ├── ✅ exp_003: Stage 3 only
│   │   ├── ⏸️ exp_004: Stage 4 only
│   │   ├── ⏸️ exp_005: Stages 2+3
│   │   ├── ⏸️ exp_006: Stages 3+4
│   │   └── ⏸️ exp_007: All stages
│   │
│   └── Phase 2: Threshold Tuning (4)
│       ├── ⏸️ exp_008: Top config, threshold=0.3
│       ├── ⏸️ exp_009: Top config, threshold=0.7
│       ├── ⏸️ exp_010: 2nd config, threshold=0.3
│       └── ⏸️ exp_011: 2nd config, threshold=0.7
│
└── GAFF Ablations (32)
    ├── Phase 1: Stage Selection (8)
    │   ├── ✅ exp_001: Stage 1 only
    │   ├── ✅ exp_002: Stage 2 only
    │   ├── ✅ exp_003: Stage 3 only
    │   ├── ✅ exp_004: Stage 4 only
    │   ├── ⏸️ exp_005: Stages 2+3
    │   ├── ⏸️ exp_006: Stages 3+4
    │   ├── ⏸️ exp_007: Stages 2+3+4
    │   └── ⏸️ exp_008: All stages
    │
    └── Phase 2: Hyperparameter Tuning (24)
        ├── ⏸️ exp_009-015: Top config × 7 hyperparam variants
        ├── ⏸️ exp_016-022: 2nd config × 7 hyperparam variants
        └── ⏸️ exp_023-032: 3rd config × 7 hyperparam variants
```

### 4.2 Theoretical Maximum Space (440 Experiments)

If exploring all combinations without constraint:

```
Dimensions:
├── Backbones: 5 (mit_b0, b1, b2, b4, b5)
├── Fusion Types: 2 (CSSA, GAFF)
├── Stage Configurations: 8 ([1], [2], [3], [4], [2,3], [3,4], [2,3,4], [1,2,3,4])
│
├── CSSA-specific:
│   ├── Thresholds: 3 (0.3, 0.5, 0.7)
│   └── Kernel sizes: 1 (3)
│   └── Total CSSA: 5 × 8 × 3 = 120 experiments
│
└── GAFF-specific:
    ├── SE Reduction: 2 (4, 8)
    ├── Inter-Modality Shared: 2 (False, True)
    └── Merge Bottleneck: 2 (False, True)
    └── Total GAFF: 5 × 8 × 2 × 2 × 2 = 320 experiments

TOTAL THEORETICAL: 120 + 320 = 440 experiments
```

**Reduction Strategy:**
1. **Fix backbone to mit_b1** for ablations → Reduces by 80%
2. **Two-phase approach:** Stage selection first, then hyperparameter tuning on top configs → Reduces search space
3. **Smart threshold/hyperparam selection:** Test extremes + default, not full grid → Further reduction

**Result:** 440 → 48 experiments (89% reduction while maintaining coverage)

---

## 5. Progress Tracking

### Overall Status
- **Total Experiments:** 48
- **Completed:** 9 (18.75%)
- **In Progress:** 1 (2.08%)
- **Pending:** 36 (75.00%)
- **Failed:** 2 (4.17%)

### By Category
| Category | Total | Complete | In Progress | Pending | Failed | Completion % |
|----------|-------|----------|-------------|---------|--------|--------------|
| Baseline | 5 | 2 | 1 | 0 | 2 | 40% |
| CSSA | 11 | 3 | 0 | 8 | 0 | 27.3% |
| GAFF | 32 | 4 | 0 | 28 | 0 | 12.5% |

### Timeline Visualization (ASCII Gantt)

```
Nov 7, 2025
├─────────────────────────────────────────────────────────────┤
│ Baseline Training                                           │
│ ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 40% (2/5 + 1 active)│
├─────────────────────────────────────────────────────────────┤
│ CSSA Ablations                                              │
│ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 27.3% (3/11)        │
├─────────────────────────────────────────────────────────────┤
│ GAFF Ablations                                              │
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 12.5% (4/32)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Naming Conventions

### CSSA Experiments
**Format:** `exp_<XXX>_s<stage_label>_t<threshold>`

**Examples:**
- `exp_001_s1_t0.5` → Experiment 001, Stage 1 only, threshold=0.5
- `exp_005_s23_t0.5` → Experiment 005, Stages 2+3, threshold=0.5
- `exp_008_s3_t0.3` → Experiment 008, Stage 3 only, threshold=0.3

**Stage Labels:**
- `s1` = [1]
- `s2` = [2]
- `s3` = [3]
- `s4` = [4]
- `s23` = [2, 3]
- `s34` = [3, 4]
- `s234` = [2, 3, 4]
- `s1234` = [1, 2, 3, 4]

### GAFF Experiments
**Format:** `exp_<XXX>_s<stage_label>_r<se_reduction>_is<inter_shared>_mb<merge_bottleneck>`

**Examples:**
- `exp_001_s1_r4_is0_mb0` → Exp 001, Stage 1, SE_r=4, not shared, no bottleneck
- `exp_009_s3_r8_is1_mb1` → Exp 009, Stage 3, SE_r=8, shared, with bottleneck
- `exp_015_s23_r4_is0_mb1` → Exp 015, Stages 2+3, SE_r=4, not shared, with bottleneck

**Parameter Encoding:**
- `r4` = SE reduction ratio 4
- `r8` = SE reduction ratio 8
- `is0` = inter_shared=False
- `is1` = inter_shared=True
- `mb0` = merge_bottleneck=False
- `mb1` = merge_bottleneck=True

---

## 7. Evaluation Metrics

All experiments are evaluated using COCO object detection metrics:

**Primary Metrics:**
- **mAP** (mean Average Precision) - Overall detection quality
- **mAP@50** - AP at IoU threshold 0.5
- **mAP@75** - AP at IoU threshold 0.75

**Object Size Metrics:**
- **mAP_small** - AP for small objects (area < 32²)
- **mAP_medium** - AP for medium objects (32² < area < 96²)
- **mAP_large** - AP for large objects (area > 96²)

**Training Metrics:**
- Training loss (per batch, per epoch)
- Validation loss
- Best epoch
- Total training time

**Model Metrics:**
- Total parameters (millions)
- Trainable parameters
- FLOPs (floating point operations)
- GPU memory consumption

---

## 8. Next Steps & Recommendations

### Immediate Priorities
1. **Complete CSSA Phase 1** (4 remaining experiments: exp_004-007)
2. **Complete GAFF Phase 1** (4 remaining experiments: exp_005-008)
3. **Resolve mit_b2 training** (currently 1/15 epochs, monitor progress)

### Memory Optimization for Large Backbones
To enable mit_b4 and mit_b5 training:
- ✅ Already implemented: Gradient accumulation (4 steps)
- ✅ Already implemented: Reduced batch size (4 vs 16)
- 🔲 Not yet tested: Gradient checkpointing (`use_grad_checkpoint=True`)
- 🔲 Alternative: Mixed precision training (FP16/BF16)
- 🔲 Alternative: Reduce FPN channels (256 → 128)

### Phase 2 Planning
- Wait for Phase 1 completion before starting Phase 2
- Identify top 2-3 performing configurations
- Launch hyperparameter tuning experiments
- Consider parallel execution for independent experiments

### Resource Estimation
**Completed experiments:**
- mit_b0: 2.68 hours
- mit_b1: 3.44 hours
- Total so far: ~6.12 hours for 2 complete baseline runs

**Estimated remaining (conservative):**
- CSSA remaining (8): ~8 × 3 hours = 24 hours
- GAFF remaining (28): ~28 × 3 hours = 84 hours
- **Total estimated:** ~108 GPU hours (4.5 days of continuous training)

### Documentation Updates
This document should be updated:
- After each experiment completes
- When Phase 1 results determine Phase 2 configurations
- When memory optimizations are tested
- When final results and conclusions are available

---

## Appendix: Quick Reference Tables

### Status Legend
- ✅ **Complete** - Experiment finished successfully with results
- 🔄 **In Progress** - Currently training
- ⏸️ **Pending** - Not yet started
- ❌ **Failed** - Terminated with error (e.g., OOM)

### File Locations
- **Training Logs:** `training_logs/run_YYYYMMDD_HHMMSS/`
- **CSSA Results:** `results/cssa_ablations/exp_XXX_*/`
- **GAFF Results:** `results/gaff_ablations_full/exp_XXX_*/`
- **Checkpoints:** `checkpoints/<backbone>/` or `<exp_dir>/checkpoints/`
- **Configuration:** `crossattentiondet/config.py`
- **Training Scripts:** `scripts/train*.py`, `crossattentiondet/ablations/scripts/*.py`

### Key Scripts
- `scripts/train.py` - Single backbone training
- `scripts/train_all_backbones_comprehensive.py` - All backbones with logging
- `scripts/resume_large_backbones.py` - Memory-optimized training for mit_b2/b4/b5
- `crossattentiondet/ablations/scripts/run_cssa_ablations.py` - CSSA ablation framework
- `crossattentiondet/ablations/scripts/run_gaff_ablations.py` - GAFF ablation framework

---

**Document End**
