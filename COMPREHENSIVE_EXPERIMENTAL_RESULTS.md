# Comprehensive Experimental Results Report

**Generated:** 2025-11-10 15:32:47

---

## Executive Summary

**Total Experiments Completed:** 39
- GAFF Ablations: 17
- CSSA Ablations (Phase 1): 8
- CSSA Ablations (Phase 2): 5
- Modality Ablations: 3
- GAFF Pilot: 1
- Baseline Models: 5

---

## 1. GAFF Ablation Study

**Fusion Method:** GAFF (Gated Attention Feature Fusion)

**Objective:** Evaluate different stage configurations and hyperparameters for GAFF fusion

### Phase 1: Stage Selection (8 experiments)

Default hyperparameters: SE_reduction=4, inter_shared=False, merge_bottleneck=False

| Exp ID | Stages | mAP | mAP@50 | mAP@75 | Train Loss | Time (h) | Params (M) |
|--------|--------|-----|--------|--------|------------|----------|------------|
| exp_001_s1_r4_is0_mb0 | 1 | 0.8274 | 0.9828 | 0.9519 | 0.0650 | 3.84 | 62.71 |
| exp_002_s2_r4_is0_mb0 | 2 | 0.8187 | 0.9818 | 0.9513 | 0.0659 | 4.06 | 62.71 |
| exp_003_s3_r4_is0_mb0 | 3 | 0.8320 | 0.9819 | 0.9525 | 0.0719 | 4.02 | 62.71 |
| exp_004_s4_r4_is0_mb0 | 4 | 0.8341 | 0.9822 | 0.9520 | 0.0747 | 4.07 | 62.71 |
| exp_005_s23_r4_is0_mb0 | 2-3 | 0.8220 | 0.9736 | 0.9499 | 0.0738 | 3.90 | 62.71 |
| exp_006_s34_r4_is0_mb0 | 3-4 | 0.8273 | 0.9819 | 0.9520 | 0.0833 | 3.91 | 62.71 |
| exp_007_s234_r4_is0_mb0 | 2-3-4 | 0.8289 | 0.9822 | 0.9507 | 0.0865 | 3.73 | 62.71 |
| exp_008_s1234_r4_is0_mb0 | 1-2-3-4 | 0.8193 | 0.9823 | 0.9524 | 0.0887 | 3.70 | 62.71 |

**Best Phase 1 Result:** exp_004_s4_r4_is0_mb0 (Stages: 4) with mAP=0.8341

### Phase 2: Hyperparameter Tuning

Tuning SE_reduction, inter_shared, and merge_bottleneck parameters

| Exp ID | Stages | SE_r | Inter_S | Merge_B | mAP | mAP@50 | mAP@75 | Loss | Time (h) | Params (M) |
|--------|--------|------|---------|---------|-----|--------|--------|------|----------|------------|
| exp_009_s4_r4_is0_mb1 | 4 | 4 | False | True | 0.8277 | 0.9746 | 0.9515 | 0.0647 | 4.13 | 63.10 |
| exp_010_s4_r4_is1_mb0 | 4 | 4 | True | False | 0.8215 | 0.9820 | 0.9523 | 0.0829 | 0.88 | 62.71 |
| exp_011_s4_r4_is1_mb1 | 4 | 4 | True | True | 0.8165 | 0.9743 | 0.9523 | 0.0650 | 4.08 | 63.10 |
| exp_012_s4_r8_is0_mb0 | 4 | 8 | False | False | 0.8378 | 0.9824 | 0.9527 | 0.0755 | 4.12 | 62.52 |
| exp_013_s4_r8_is0_mb1 | 4 | 8 | False | True | 0.8253 | 0.9741 | 0.9515 | 0.0667 | 4.08 | 62.91 |
| exp_014_s4_r8_is1_mb0 | 4 | 8 | True | False | 0.8320 | 0.9831 | 0.9526 | 0.0753 | 4.21 | 62.52 |
| exp_015_s4_r8_is1_mb1 | 4 | 8 | True | True | 0.8336 | 0.9746 | 0.9513 | 0.0671 | 4.19 | 62.91 |
| exp_016_s3_r4_is0_mb1 | 3 | 4 | False | True | 0.8338 | 0.9814 | 0.9601 | 0.0648 | 4.17 | 63.10 |
| exp_017_s3_r4_is1_mb0 | 3 | 4 | True | False | 0.8246 | 0.9827 | 0.9520 | 0.0705 | 4.09 | 62.71 |

**Best Overall GAFF Result:** exp_012_s4_r8_is0_mb0 (Stages: 4, SE_r=8, Inter_S=False, Merge_B=False) with mAP=0.8378

### GAFF Detailed Metrics

Additional metrics from completed experiments:

| Exp ID | mAP_small | mAP_medium | mAP_large | Epochs | Batch Size | LR |
|--------|-----------|------------|-----------|--------|------------|-----|
| exp_001_s1_r4_is0_mb0 | 0.6418 | 0.8405 | 0.7651 | 15 | 16 | 0.02 |
| exp_002_s2_r4_is0_mb0 | 0.6388 | 0.8312 | 0.8049 | 15 | 16 | 0.02 |
| exp_003_s3_r4_is0_mb0 | 0.6517 | 0.8464 | 0.7663 | 15 | 16 | 0.02 |
| exp_004_s4_r4_is0_mb0 | 0.6430 | 0.8496 | 0.7529 | 15 | 16 | 0.02 |
| exp_005_s23_r4_is0_mb0 | 0.6245 | 0.8371 | 0.7822 | 15 | 16 | 0.02 |
| exp_006_s34_r4_is0_mb0 | 0.6323 | 0.8391 | 0.8753 | 15 | 16 | 0.02 |
| exp_007_s234_r4_is0_mb0 | 0.6222 | 0.8421 | 0.8307 | 15 | 16 | 0.02 |
| exp_008_s1234_r4_is0_mb0 | 0.6267 | 0.8307 | 0.7058 | 15 | 16 | 0.02 |
| exp_009_s4_r4_is0_mb1 | 0.6107 | 0.8435 | 0.7446 | 15 | 16 | 0.02 |
| exp_010_s4_r4_is1_mb0 | 0.6262 | 0.8347 | 0.7751 | 15 | 16 | 0.02 |
| exp_011_s4_r4_is1_mb1 | 0.6072 | 0.8326 | 0.8361 | 15 | 16 | 0.02 |
| exp_012_s4_r8_is0_mb0 | 0.6452 | 0.8530 | 0.8522 | 15 | 16 | 0.02 |
| exp_013_s4_r8_is0_mb1 | 0.6161 | 0.8416 | 0.8259 | 15 | 16 | 0.02 |
| exp_014_s4_r8_is1_mb0 | 0.6301 | 0.8475 | 0.8548 | 15 | 16 | 0.02 |
| exp_015_s4_r8_is1_mb1 | 0.6369 | 0.8483 | 0.8172 | 15 | 16 | 0.02 |
| exp_016_s3_r4_is0_mb1 | 0.6521 | 0.8463 | 0.7492 | 15 | 16 | 0.02 |
| exp_017_s3_r4_is1_mb0 | 0.6309 | 0.8389 | 0.8487 | 15 | 16 | 0.02 |
| exp_020_s3_r8_is0_mb1 | 0.6302 | 0.8451 | 0.7884 | 15 | 16 | 0.02 |

---

## 2. CSSA Ablation Study - Phase 1

**Fusion Method:** CSSA (Channel Switching and Spatial Attention)

**Objective:** Evaluate CSSA fusion at different backbone stages with threshold=0.5

| Exp ID | Stages | Threshold | mAP | mAP@50 | mAP@75 | mAP_small | mAP_medium | mAP_large | Loss | Time (h) | Params (M) |
|--------|--------|-----------|-----|--------|--------|-----------|------------|-----------|------|----------|------------|
| exp_001_s1_t0.5 | [1] | N/A | 0.8407 | 0.9737 | 0.9513 | 0.6290 | 0.8564 | 0.8007 | 0.0437 | 14.83 | 60.01 |
| exp_001_s1_t0.5_fast | [1] | N/A | 0.8344 | 0.9745 | 0.9514 | 0.6460 | 0.8489 | 0.8450 | 0.0700 | 3.77 | 60.01 |
| exp_002_s2_t0.5 | [2] | N/A | 0.8258 | 0.9742 | 0.9526 | 0.6296 | 0.8418 | 0.7666 | 0.0706 | 3.89 | 60.01 |
| exp_003_s3_t0.5 | [3] | N/A | 0.8280 | 0.9734 | 0.9502 | 0.6288 | 0.8432 | 0.7646 | 0.0726 | 3.84 | 60.01 |
| exp_004_s4_t0.5 | [4] | N/A | 0.8320 | 0.9828 | 0.9514 | 0.6229 | 0.8472 | 0.7861 | 0.0869 | 4.03 | 60.01 |
| exp_005_s23_t0.5 | [2, 3] | N/A | 0.8232 | 0.9820 | 0.9512 | 0.6299 | 0.8374 | 0.8624 | 0.0801 | 3.95 | 60.01 |
| exp_006_s34_t0.5 | [3, 4] | N/A | 0.8166 | 0.9822 | 0.9518 | 0.6295 | 0.8280 | 0.6218 | 0.0967 | 3.90 | 60.01 |
| exp_007_s1234_t0.5 | [1, 2, 3, 4] | N/A | 0.8091 | 0.9803 | 0.9389 | 0.5935 | 0.8250 | 0.6561 | 0.1139 | 3.54 | 60.01 |

**Best CSSA Phase 1 Result:** exp_001_s1_t0.5 (Stages: [1]) with mAP=0.8407

---

## 3. CSSA Ablation Study - Phase 2

**Objective:** Fine-tune CSSA threshold parameter on best stage configurations

| Exp ID | Stages | Threshold | mAP | mAP@50 | mAP@75 | mAP_small | mAP_medium | mAP_large | Loss | Time (h) | Params (M) |
|--------|--------|-----------|-----|--------|--------|-----------|------------|-----------|------|----------|------------|
| exp_008_s1_t0.3 | [1] | N/A | 0.8323 | 0.9746 | 0.9522 | 0.6352 | 0.8476 | 0.8476 | 0.0671 | 3.73 | 60.01 |
| exp_009_s1_t0.7 | [1] | N/A | 0.8298 | 0.9744 | 0.9521 | 0.6145 | 0.8453 | 0.8344 | 0.0695 | 3.98 | 60.01 |
| exp_010_s4_t0.3 | [4] | N/A | 0.8309 | 0.9819 | 0.9587 | 0.6333 | 0.8443 | 0.7648 | 0.0895 | 4.03 | 60.01 |
| exp_011_s3_t0.3 | [3] | N/A | 0.7983 | 0.9714 | 0.9389 | 0.5523 | 0.8170 | 0.7004 | 0.0769 | 3.98 | 60.01 |
| exp_012_s23_t0.3 | [2, 3] | N/A | 0.8170 | 0.9737 | 0.9493 | 0.6070 | 0.8324 | 0.7461 | 0.0786 | 3.96 | 60.01 |

**Best CSSA Phase 2 Result:** exp_008_s1_t0.3 (Stages: [1], Threshold: N/A) with mAP=0.8323

---

## 4. Baseline Models (15 epochs) - Modality Combinations

**Architecture:** Standard CMX with FRM+FFM (baseline_FRM_FFM)

**Training:** 15 epochs, batch_size=16, lr=0.02, backbone=mit_b1

**Objective:** Establish baseline performance with standard fusion across different modality pairs

**⚠️ IMPORTANT:** These are BASELINE experiments, not ablations. They use the standard CMX architecture.

| Modality Pair | mAP | mAP@50 | mAP@75 | mAP_small | mAP_medium | mAP_large | Loss | Time (h) | Params (M) | Backbone |
|---------------|-----|--------|--------|-----------|------------|-----------|------|----------|------------|----------|
| RGB + EVENT | 0.6632 | 0.9446 | 0.7522 | 0.3314 | 0.6903 | 0.6569 | 0.1592 | 4.10 | 60.01 | mit_b1 |
| RGB + THERMAL | 0.8342 | 0.9822 | 0.9602 | 0.6695 | 0.8458 | 0.7834 | 0.0946 | 4.20 | 60.01 | mit_b1 |
| THERMAL + EVENT | 0.7486 | 0.9695 | 0.8929 | 0.6108 | 0.7584 | 0.8399 | 0.1192 | 4.08 | 60.01 | mit_b1 |

**Best Baseline (15 epochs):** RGB + THERMAL with mAP=0.8342 (83.42%)

**This is the primary baseline for comparison with GAFF and CSSA ablations.**

---

## 5. GAFF Pilot Study

**Objective:** 5-epoch pilot to validate GAFF implementation

| Metric | Value |
|--------|-------|
| Experiment ID | gaff_pilot_5epochs |
| Epochs | 5 |
| Backbone | mit_b1 |
| GAFF Stages | [4] |
| mAP | 0.7946 |
| mAP@50 | 0.9788 |
| mAP@75 | 0.9462 |
| mAP Small | 0.6064 |
| mAP Medium | 0.8086 |
| mAP Large | 0.6761 |
| Final Train Loss | 0.1225 |
| Training Time (hours) | 2.57 |
| Total Parameters (M) | 62.71 |

---

## 6. Baseline Models (1 epoch) - Initial Validation

**Fusion Method:** Original CMX with Feature Rectify Module (FRM) and Feature Fusion Module (FFM)

**Training:** 1 epoch, batch_size=2, lr=0.005

**Important Note:** These baseline models were trained for only 1 epoch for initial validation. 
For fair comparison with ablation studies (15 epochs), these would need to be retrained.

| Backbone | Params (M) | mAP | mAP@50 | Size (MB) | Inference (ms) | FPS | Training (min) |
|----------|------------|-----|--------|-----------|----------------|-----|----------------|
| mit_b1 | 60.0 | 0.6624 | 0.9490 | 228.95 | 54.54 | 18.34 | 26.42 |
| mit_b0 | 27.8 | 0.6465 | 0.9391 | 106.00 | 42.59 | 23.48 | 22.91 |
| mit_b5 | 196.6 | 0.6318 | 0.9402 | 749.98 | 175.21 | 5.71 | 74.35 |
| mit_b4 | 155.4 | 0.6270 | 0.9315 | 592.81 | 145.83 | 6.86 | 64.87 |
| mit_b2 | 82.1 | 0.6242 | 0.9440 | 313.22 | 78.57 | 12.73 | 37.96 |

**Best Baseline (1 epoch):** mit_b1 with mAP=0.6624 (66.24%)

### Performance vs Ablation Studies

**⚠️ See Section 4 for fair comparison:** The 15-epoch baseline (RGB+Thermal) achieves 83.42% mAP.

**1-epoch results (this section):**
- Best 1-epoch baseline: **66.24% mAP** (mit_b1)

**15-epoch results (Section 4):**
- Best 15-epoch baseline: **83.42% mAP** (RGB+Thermal)
- Best GAFF ablation: **83.78% mAP** (+0.36%)
- Best CSSA ablation: **84.07% mAP** (+0.65%)

### Speed-Accuracy Trade-off

**Fastest:** mit_b0 (23.48 FPS, mAP=0.6465)

**Best Accuracy:** mit_b1 (18.34 FPS, mAP=0.6624)

**Slowest:** mit_b5 (5.71 FPS, mAP=0.6318)

---
## 7. Key Findings

### Top 10 Performing Experiments (by mAP)

| Rank | Study | Experiment | mAP | Δ vs Baseline |
|------|-------|------------|-----|---------------|
| 1 | CSSA-P1 | exp_001_s1_t0.5 | 0.8407 | +0.65% |
| 2 | GAFF | exp_012_s4_r8_is0_mb0 | 0.8378 | +0.36% |
| 3 | CSSA-P1 | exp_001_s1_t0.5_fast | 0.8344 | +0.02% |
| **4** | **Baseline** | **rgb_thermal (FRM+FFM)** | **0.8342** | **—** |
| 5 | GAFF | exp_004_s4_r4_is0_mb0 | 0.8341 | -0.01% |
| 6 | GAFF | exp_016_s3_r4_is0_mb1 | 0.8338 | -0.04% |
| 7 | GAFF | exp_015_s4_r8_is1_mb1 | 0.8336 | -0.06% |
| 8 | CSSA-P2 | exp_008_s1_t0.3 | 0.8323 | -0.19% |
| 9 | GAFF | exp_014_s4_r8_is1_mb0 | 0.8320 | -0.22% |
| 10 | CSSA-P1 | exp_004_s4_t0.5 | 0.8320 | -0.22% |

### Critical Findings: Baseline vs Ablations

**Baseline Performance (15 epochs, RGB+Thermal):** 83.42% mAP

**Ablation Study Results:**
- Best GAFF: 83.78% mAP (+0.36% vs baseline)
- Best CSSA: 84.07% mAP (+0.65% vs baseline)
- Best CSSA fast: 83.44% mAP (+0.02% vs baseline)

**Key Insight:** The fusion ablation methods (GAFF, CSSA) provide **marginal improvements (~0.5%)** over the standard FRM+FFM baseline when trained under identical conditions (15 epochs, same hyperparameters).

### Insights

#### GAFF Ablations:
- Total configurations tested: 17
- mAP range: 0.8165 to 0.8378
- Average mAP: 0.8272

#### CSSA Ablations:
- Total configurations tested: 13
- mAP range: 0.7983 to 0.8407
- Average mAP: 0.8245

#### Baseline Modality Combinations:
- Total combinations tested: 3
- mAP range: 0.6632 to 0.8342
- Average mAP: 0.7487
- **Best baseline: RGB+Thermal at 83.42% mAP**

**Modality Importance:**
- RGB alone: Not tested separately
- RGB + Thermal: 83.42% mAP (best combination)
- RGB + Event: 66.32% mAP (poor performance)
- Thermal + Event: 74.86% mAP (moderate)
- **Conclusion:** RGB+Thermal is the optimal modality pair

---

## 8. Experimental Setup

### Common Configuration

- **Dataset:** RGBX Object Detection
- **Training Images:** 7,800
- **Test Images:** 1,950
- **Batch Size:** 16 (most experiments)
- **Learning Rate:** 0.02
- **Epochs:** 15 (main studies), 5 (pilot)
- **Optimizer:** SGD with momentum
- **Backbone:** MixTransformer (mit_b1, mit_b0)

### Fusion Methods Evaluated

1. **GAFF (Gated Attention Feature Fusion)**
   - Gated attention mechanism for multi-modal fusion
   - Tested at different encoder stages (1, 2, 3, 4, and combinations)
   - Hyperparameters: SE reduction ratio, inter-modality sharing, merge bottleneck

2. **CSSA (Channel Switching and Spatial Attention)**
   - Channel-wise attention with threshold-based switching
   - Spatial attention using max and average pooling
   - Tested thresholds: 0.3, 0.5, 0.7

---

## Notes

- All experiments used CUDA GPU acceleration
- Training time varies based on stage configuration and hyperparameters
- Results are reported on the test set after final epoch
- mAP metrics follow COCO evaluation protocol

---

*Report generated on 2025-11-10 at 15:32:47*