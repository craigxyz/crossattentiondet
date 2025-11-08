# Ablation Studies Framework

**Comprehensive 48-Experiment Ablation Design**

[← Back to Index](00_INDEX.md) | [← Previous: Dataset](03_DATASET_AND_MODALITIES.md) | [Next: Training & Hyperparameters →](05_TRAINING_AND_HYPERPARAMETERS.md)

---

## Quick Reference

**Total Experiments:** 48
- Baseline Backbones: 5
- CSSA Ablations: 11  
- GAFF Ablations: 32

**Current Status:** 9/48 complete (18.75%)

**Methodology:** Two-phase (Stage Selection → Hyperparameter Tuning)

---

## Complete Ablation Matrix

### Phase 1: Stage Selection

**CSSA (7 experiments):**
| Exp ID | Stages | Threshold | Status |
|--------|--------|-----------|--------|
| exp_001 | [1] | 0.5 | ✅ Complete |
| exp_002 | [2] | 0.5 | ✅ Complete |
| exp_003 | [3] | 0.5 | ✅ Complete |
| exp_004 | [4] | 0.5 | ⏸️ Pending |
| exp_005 | [2,3] | 0.5 | ⏸️ Pending |
| exp_006 | [3,4] | 0.5 | ⏸️ Pending |
| exp_007 | [1,2,3,4] | 0.5 | ⏸️ Pending |

**GAFF (8 experiments):**
| Exp ID | Stages | SE_r | inter_shared | merge_btlnk | Status |
|--------|--------|------|--------------|-------------|--------|
| exp_001 | [1] | 4 | False | False | ✅ Complete |
| exp_002 | [2] | 4 | False | False | ✅ Complete |
| exp_003 | [3] | 4 | False | False | ✅ Complete |
| exp_004 | [4] | 4 | False | False | ✅ Complete |
| exp_005 | [2,3] | 4 | False | False | ⏸️ Pending |
| exp_006 | [3,4] | 4 | False | False | ⏸️ Pending |
| exp_007 | [2,3,4] | 4 | False | False | ⏸️ Pending |
| exp_008 | [1,2,3,4] | 4 | False | False | ⏸️ Pending |

### Phase 2: Hyperparameter Tuning

**CSSA (4 experiments):**
- Select top 2 stage configs from Phase 1
- Test thresholds: {0.3, 0.7} (already tested 0.5)

**GAFF (24 experiments):**
- Select top 3 stage configs from Phase 1
- Test grid: SE_reduction × inter_shared × merge_bottleneck
  - SE_reduction: {4, 8}
  - inter_shared: {False, True}
  - merge_bottleneck: {False, True}
- Total: 3 configs × (2×2×2) = 24 experiments

---

## Hypothesis & Expected Results

**Best Single Stage:** Stage 3 (320 channels, 14×14, mid-level semantics)

**Best Multi-Stage:** [2,3] or [3,4] (avoid stage 1 overhead)

**CSSA vs. GAFF:** +2-3% mAP for GAFF, +355× parameters

---

For full details, see:
- `docs/CSSA_ABLATION_GUIDE.md` (791 lines)
- `docs/GAFF_ABLATION_GUIDE.md` (1053 lines)
- `docs/EXPERIMENTAL_MATRIX.md`

[← Back to Index](00_INDEX.md) | [← Previous: Dataset](03_DATASET_AND_MODALITIES.md) | [Next: Training & Hyperparameters →](05_TRAINING_AND_HYPERPARAMETERS.md)
