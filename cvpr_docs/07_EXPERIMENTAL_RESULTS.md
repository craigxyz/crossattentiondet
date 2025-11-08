# Experimental Results

**Current Results and Progress Tracking**

[← Back to Index](00_INDEX.md) | [← Previous: Implementation](06_IMPLEMENTATION_DETAILS.md) | [Next: CVPR Paper Guidance →](08_CVPR_PAPER_GUIDANCE.md)

---

## Current Status

**Total Experiments:** 48
**Completed:** 9 (18.75%)
**In Progress:** 1 (mit_b2 backbone)
**Remaining:** 38

---

## Baseline Backbone Results

| Backbone | Params | Epochs | Best Loss | Training Time | Status |
|----------|--------|--------|-----------|---------------|--------|
| mit_b0 | 55.7M | 15/15 | 0.1057 | 2.68h | ✅ Complete |
| **mit_b1** | **69.5M** | **15/15** | **0.1027** | **3.44h** | ✅ **BEST** |
| mit_b2 | 82.1M | 1/15 | 0.3787 | ~6.75h (est) | 🔄 Training |
| mit_b4 | 155.4M | 0/15 | - | - | ❌ CUDA OOM |
| mit_b5 | 196.6M | 0/15 | - | - | ❌ CUDA OOM |

**Key Finding:** mit_b1 provides best accuracy/efficiency trade-off

---

## CSSA Ablation Results

**Completed:** 3/11 (27.3%)

| Exp | Stages | Threshold | Status | Results |
|-----|--------|-----------|--------|---------|
| 001 | [1] | 0.5 | ✅ | Check `results/cssa_ablations/exp_001_s1_t0.5/final_results.json` |
| 002 | [2] | 0.5 | ✅ | Check `results/cssa_ablations/exp_002_s2_t0.5/final_results.json` |
| 003 | [3] | 0.5 | ✅ | Check `results/cssa_ablations/exp_003_s3_t0.5/final_results.json` |

---

## GAFF Ablation Results

**Completed:** 4/32 (12.5%)

| Exp | Stages | Config | Status | Results |
|-----|--------|--------|--------|---------|
| 001 | [1] | r4_is0_mb0 | ✅ | Check `results/gaff_ablations_full/phase1_stage_selection/exp_001.../` |
| 002 | [2] | r4_is0_mb0 | ✅ | Check phase1 directory |
| 003 | [3] | r4_is0_mb0 | ✅ | Check phase1 directory |
| 004 | [4] | r4_is0_mb0 | ✅ | Check phase1 directory |

---

## Logging Infrastructure

**Per-Experiment Output (13 files):**
1. training.log
2. metrics_per_epoch.csv
3. metrics_per_batch.csv
4. evaluation_history.json
5. final_results.json
6. config.json
7. model_info.json
8-10. checkpoint_epoch_{5,10,15}.pth
11. checkpoint_latest.pth
12. checkpoint_best.pth
13. model_best_weights.pth

**Storage:** ~500 MB per experiment

---

## Expected Results (Hypothesis)

| Configuration | Expected mAP | Params Overhead | Deployment |
|---------------|--------------|-----------------|------------|
| Baseline | 40-50% | 0% | - |
| CSSA (best) | 42-52% | +0.003% | Edge |
| GAFF (best) | 43-54% | +1-3% | Cloud |

**Validation pending completion of all experiments.**

---

[← Back to Index](00_INDEX.md) | [← Previous: Implementation](06_IMPLEMENTATION_DETAILS.md) | [Next: CVPR Paper Guidance →](08_CVPR_PAPER_GUIDANCE.md)
