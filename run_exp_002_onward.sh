#!/bin/bash

# Script to run CSSA ablation experiments 002-007 (Phase 1 remaining)
# Run this while exp_001 finishes in another terminal

set -e  # Exit on error

# Configuration
DATA_DIR="../RGBX_Semantic_Segmentation/data/images"
LABELS_DIR="../RGBX_Semantic_Segmentation/data/labels"
OUTPUT_BASE="results/cssa_ablations"
BACKBONE="mit_b1"
EPOCHS=15
BATCH_SIZE=16
LR=0.02
THRESHOLD=0.5
KERNEL=3

echo "================================================================================"
echo "  CSSA Ablation Study - Experiments 002-007"
echo "================================================================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Experiment 002: Stage 2
echo "=== Experiment 002: Stage 2, threshold=0.5 ==="
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data "$DATA_DIR" \
    --labels "$LABELS_DIR" \
    --output-dir "$OUTPUT_BASE/exp_002_s2_t0.5" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --cssa-stages "2" \
    --cssa-thresh "$THRESHOLD" \
    --cssa-kernel "$KERNEL"

echo ""
echo "✓ Experiment 002 completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Experiment 003: Stage 3
echo "=== Experiment 003: Stage 3, threshold=0.5 ==="
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data "$DATA_DIR" \
    --labels "$LABELS_DIR" \
    --output-dir "$OUTPUT_BASE/exp_003_s3_t0.5" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --cssa-stages "3" \
    --cssa-thresh "$THRESHOLD" \
    --cssa-kernel "$KERNEL"

echo ""
echo "✓ Experiment 003 completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Experiment 004: Stage 4
echo "=== Experiment 004: Stage 4, threshold=0.5 ==="
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data "$DATA_DIR" \
    --labels "$LABELS_DIR" \
    --output-dir "$OUTPUT_BASE/exp_004_s4_t0.5" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --cssa-stages "4" \
    --cssa-thresh "$THRESHOLD" \
    --cssa-kernel "$KERNEL"

echo ""
echo "✓ Experiment 004 completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Experiment 005: Stages 2+3
echo "=== Experiment 005: Stages 2+3, threshold=0.5 ==="
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data "$DATA_DIR" \
    --labels "$LABELS_DIR" \
    --output-dir "$OUTPUT_BASE/exp_005_s23_t0.5" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --cssa-stages "2,3" \
    --cssa-thresh "$THRESHOLD" \
    --cssa-kernel "$KERNEL"

echo ""
echo "✓ Experiment 005 completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Experiment 006: Stages 3+4
echo "=== Experiment 006: Stages 3+4, threshold=0.5 ==="
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data "$DATA_DIR" \
    --labels "$LABELS_DIR" \
    --output-dir "$OUTPUT_BASE/exp_006_s34_t0.5" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --cssa-stages "3,4" \
    --cssa-thresh "$THRESHOLD" \
    --cssa-kernel "$KERNEL"

echo ""
echo "✓ Experiment 006 completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Experiment 007: All stages
echo "=== Experiment 007: All stages (1,2,3,4), threshold=0.5 ==="
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    --data "$DATA_DIR" \
    --labels "$LABELS_DIR" \
    --output-dir "$OUTPUT_BASE/exp_007_s1234_t0.5" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --cssa-stages "1,2,3,4" \
    --cssa-thresh "$THRESHOLD" \
    --cssa-kernel "$KERNEL"

echo ""
echo "✓ Experiment 007 completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "================================================================================"
echo "  ALL EXPERIMENTS 002-007 COMPLETED!"
echo "================================================================================"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Next: Analyze results with:"
echo "  python crossattentiondet/ablations/scripts/analyze_cssa_results.py --results-dir results/cssa_ablations"
