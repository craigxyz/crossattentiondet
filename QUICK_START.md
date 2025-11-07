# CSSA Ablation - Quick Start Guide

## 🚀 Run Full Ablation Study (Recommended)

```bash
python -u crossattentiondet/ablations/scripts/run_cssa_ablations.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-base results/cssa_ablations \
    --epochs 25 \
    --backbone mit_b1
```

**What it does:**
- Runs 11 experiments automatically (7 stage configs + 4 threshold variants)
- ~66-77 hours total (~3 days)
- Saves everything to `results/cssa_ablations/`

## 📊 Analyze Results

```bash
python crossattentiondet/ablations/scripts/analyze_cssa_results.py \
    --results-dir results/cssa_ablations
```

**Creates:**
- Comparison tables (CSV + markdown)
- Plots (mAP, threshold sensitivity, size-specific)
- Summary report

## 🧪 Test Single Experiment (5 epochs, ~30 min)

```bash
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/cssa_ablations/test_exp \
    --epochs 5 \
    --batch-size 2 \
    --backbone mit_b1 \
    --cssa-stages "3,4" \
    --cssa-thresh 0.5
```

## 👀 Monitor Progress

```bash
# Watch overall progress
tail -f results/cssa_ablations/master_log.txt

# Watch specific experiment
tail -f results/cssa_ablations/exp_001_s1_t0.5/training.log

# Check completed experiments
ls results/cssa_ablations/exp_*/final_results.json | wc -l
```

## 📁 Results Location

All results in: `results/cssa_ablations/`

**Key files:**
- `master_log.txt` - Overall progress
- `summary_all_experiments.csv` - All results
- `exp_XXX_*/final_results.json` - Per-experiment results
- `analysis/RESULTS_SUMMARY.md` - Final report

## ⏱️ Timeline

- Single experiment: ~6-7 hours
- Full study: ~66-77 hours (~3 days)
- Analysis: ~5 minutes

## 🎯 What You Get

**11 Experiments:**
1. Stage 1 only (early fusion)
2. Stage 2 only
3. Stage 3 only
4. Stage 4 only (late fusion)
5. Stages 2+3
6. Stages 3+4
7. All stages (progressive)
8-11. Top 2 configs at thresholds {0.3, 0.7}

**Analysis:**
- Stage comparison table
- Threshold sensitivity analysis
- mAP comparison plots
- Size-specific performance
- Publication-ready summaries

## 📖 Full Documentation

- `crossattentiondet/ablations/ABLATION_GUIDE.md` - Complete guide
- `ABLATION_SYSTEM_SUMMARY.md` - System overview

## ✅ Tested & Ready

All components tested and working:
- ✅ Flexible encoder (all stage combinations)
- ✅ Training script with logging
- ✅ Master runner
- ✅ Analysis tools
- ✅ Data paths verified

**Start your ablation study now!** 🎉
