#!/bin/bash
# Quick re-run of exp_001 with aggressive hyperparameters
# This determines if Stage 1's superiority persists with fast training

echo "========================================================================"
echo "         Re-running exp_001 with Aggressive Hyperparameters"
echo "========================================================================"
echo "Configuration:"
echo "  - Stage: 1 only"
echo "  - Epochs: 15 (vs original 25)"
echo "  - Batch Size: 16 (vs original 2)"
echo "  - Learning Rate: 0.02 (vs original 0.005)"
echo "  - Expected Time: ~3.5-4 hours (vs original 14.83 hours)"
echo ""
echo "Original exp_001 result: 84.07% mAP"
echo "Will this hold with aggressive training? Let's find out..."
echo "========================================================================"
echo ""

python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data ../RGBX_Semantic_Segmentation/data/images \
    --labels ../RGBX_Semantic_Segmentation/data/labels \
    --output-dir results/cssa_ablations/exp_001_s1_t0.5_fast \
    --backbone mit_b1 \
    --epochs 15 \
    --batch-size 16 \
    --lr 0.02 \
    --cssa-stages 1 \
    --cssa-thresh 0.5 \
    --cssa-kernel 3

echo ""
echo "========================================================================"
echo "                        EXPERIMENT COMPLETE!"
echo "========================================================================"
echo "Results saved to: results/cssa_ablations/exp_001_s1_t0.5_fast/"
echo ""
echo "Compare results:"
echo "  Original (25 epochs, bs=2):  results/cssa_ablations/exp_001_s1_t0.5/final_results.json"
echo "  Fast (15 epochs, bs=16):     results/cssa_ablations/exp_001_s1_t0.5_fast/final_results.json"
echo ""
echo "Check if Stage 1 still beats Stage 4 (83.20% mAP) with fast training!"
echo "========================================================================"
