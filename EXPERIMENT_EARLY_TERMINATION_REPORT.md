# Experiment Early Termination Report

Generated: 2025-11-10

## Summary

**Total experiments across all ablation studies:** 49 planned / 26 exist
**Completed:** 23
**Ended early:** 2
**Never started:** 11

---

## 1. GAFF Ablations (results/gaff_ablations_full/)

**Status:** 18/29 completed, 2 ended early, 9 never started

### ✓ Completed Experiments (18)

#### Phase 1: Stage Selection (8/8 completed)
- exp_001_s1_r4_is0_mb0: 15/15 epochs ✓
- exp_002_s2_r4_is0_mb0: 15/15 epochs ✓
- exp_003_s3_r4_is0_mb0: 15/15 epochs ✓
- exp_004_s4_r4_is0_mb0: 15/15 epochs ✓
- exp_005_s23_r4_is0_mb0: 15/15 epochs ✓
- exp_006_s34_r4_is0_mb0: 15/15 epochs ✓
- exp_007_s234_r4_is0_mb0: 15/15 epochs ✓
- exp_008_s1234_r4_is0_mb0: 15/15 epochs ✓

#### Phase 2: Hyperparameter Tuning (10/21 completed)
- exp_009_s4_r4_is0_mb1: 15/15 epochs ✓
- exp_010_s4_r4_is1_mb0: 15/15 epochs ✓
- exp_011_s4_r4_is1_mb1: 15/15 epochs ✓
- exp_012_s4_r8_is0_mb0: 15/15 epochs ✓
- exp_013_s4_r8_is0_mb1: 15/15 epochs ✓
- exp_014_s4_r8_is1_mb0: 15/15 epochs ✓
- exp_015_s4_r8_is1_mb1: 15/15 epochs ✓
- exp_016_s3_r4_is0_mb1: 15/15 epochs ✓
- exp_017_s3_r4_is1_mb0: 15/15 epochs ✓
- exp_020_s3_r8_is0_mb1: 15/15 epochs ✓

### ✗ ENDED EARLY (2 experiments)

**exp_018_s3_r4_is1_mb1**
- Location: `results/gaff_ablations_full/phase2_hyperparameter_tuning/exp_018_s3_r4_is1_mb1/`
- Expected: 15 epochs
- Completed: **8/15 epochs (53%)**
- Missing: 7 epochs
- Config: stages=[3], SE_reduction=4, inter_shared=True, merge_bottleneck=True
- Status: Training interrupted on 2025-11-10 during main ablation run

**exp_021_s3_r8_is1_mb0**
- Location: `results/gaff_ablations_full/phase2_hyperparameter_tuning/exp_021_s3_r8_is1_mb0/`
- Expected: 15 epochs
- Completed: **8/15 epochs (53%)**
- Missing: 7 epochs
- Config: stages=[3], SE_reduction=8, inter_shared=True, merge_bottleneck=False
- Status: Training interrupted during resume attempt

### ⚠ NEVER STARTED (9 experiments)

These experiments were planned but never executed:

**exp_019_s3_r8_is0_mb0**
- Config: stages=[3], SE_reduction=8, inter_shared=False, merge_bottleneck=False
- Part: Should have been in Part 1 (Computer 1)

**exp_022_s3_r8_is1_mb1**
- Config: stages=[3], SE_reduction=8, inter_shared=True, merge_bottleneck=True
- Part: Should have been in Part 2 (Computer 2)

**exp_023 through exp_029 (7 experiments)**
All with stages=[2,3,4], various hyperparameter combinations:
- exp_023: SE_r=4, inter_shared=False, merge_bottleneck=True
- exp_024: SE_r=4, inter_shared=True, merge_bottleneck=False
- exp_025: SE_r=4, inter_shared=True, merge_bottleneck=True
- exp_026: SE_r=8, inter_shared=False, merge_bottleneck=False
- exp_027: SE_r=8, inter_shared=False, merge_bottleneck=True
- exp_028: SE_r=8, inter_shared=True, merge_bottleneck=False
- exp_029: SE_r=8, inter_shared=True, merge_bottleneck=True

---

## 2. CSSA Ablations Phase 2 (results/cssa_ablations_phase2/)

**Status:** 5/5 completed ✓

All CSSA experiments completed successfully:

- exp_008_s1_t0.3: 15/15 epochs ✓
- exp_009_s1_t0.7: 15/15 epochs ✓
- exp_010_s4_t0.3: 15/15 epochs ✓
- exp_011_s3_t0.3: 15/15 epochs ✓
- exp_012_s23_t0.3: 15/15 epochs ✓

---

## 3. Modality Ablations (results/modality_ablations/)

**Status:** 3/3 completed ✓

All modality ablation experiments completed successfully:

- rgb_thermal: 15/15 epochs ✓ (completed 2025-11-08 13:14:16)
- thermal_event: 15/15 epochs ✓ (completed 2025-11-08 21:30:59)
- rgb_event: 15/15 epochs ✓ (completed 2025-11-08 17:23:18)

---

## 4. Other Experiments

### GAFF Pilot (gaff_pilot_5epochs/)

**Status:** Completed ✓
- Training: 5/5 epochs ✓ (completed 2025-11-07 16:38:08)
- Purpose: 5-epoch pilot study
- Result: mAP=0.7946, training time=2.57 hours

---

## Action Items

### To Resume Early-Terminated Experiments:

**exp_018_s3_r4_is1_mb1:** Resume from epoch 9, complete 7 more epochs (~2.9 hours)
```bash
python train_gaff_ablation_resume.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/gaff_ablations_full/phase2_hyperparameter_tuning/exp_018_s3_r4_is1_mb1 \
    --backbone mit_b1 \
    --epochs 15 \
    --batch-size 16 \
    --lr 0.02 \
    --gaff-stages 3 \
    --gaff-se-reduction 4 \
    --gaff-inter-shared true \
    --gaff-merge-bottleneck true \
    --resume-from-checkpoint true
```

**exp_021_s3_r8_is1_mb0:** Resume from epoch 9, complete 7 more epochs (~2.9 hours)
```bash
python train_gaff_ablation_resume.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/gaff_ablations_full/phase2_hyperparameter_tuning/exp_021_s3_r8_is1_mb0 \
    --backbone mit_b1 \
    --epochs 15 \
    --batch-size 16 \
    --lr 0.02 \
    --gaff-stages 3 \
    --gaff-se-reduction 8 \
    --gaff-inter-shared true \
    --gaff-merge-bottleneck false \
    --resume-from-checkpoint true
```

### To Run Never-Started Experiments:

The resume scripts `resume_gaff_part1.py` and `resume_gaff_part2.py` can be modified or run individually to complete the missing 9 experiments (exp_019, exp_022-029). Estimated time: ~36 hours total (4 hours per experiment).

---

## Files Generated

- `check_early_termination.py` - Script to check GAFF experiment status
- `check_all_experiments.py` - Comprehensive checker for all ablation studies
- `EXPERIMENT_EARLY_TERMINATION_REPORT.md` - This report

---

## Notes

- The original master ablation run (master_log.txt) was interrupted at exp_018
- Resume scripts (part1/part2) were created to complete remaining experiments but were not fully executed
- All CSSA and modality ablation experiments completed successfully without interruption
- The 2 early-terminated experiments both stopped at exactly 8/15 epochs, suggesting a common cause (possibly time limit or system interruption)
