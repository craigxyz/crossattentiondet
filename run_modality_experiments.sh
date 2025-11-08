#!/bin/bash
#
# Launch script for modality ablation experiments
# Runs 3 experiments in sequence: RGB+Thermal, RGB+Event, Thermal+Event
#
# Configuration optimized for A100 GPU:
# - Batch size: 32
# - Learning rate: 0.04 (scaled with batch size)
# - Epochs: 15
# - Backbone: mit_b1
# - Architecture: Baseline FRM+FFM (NOT GAFF/CSSA)
#

set -e  # Exit on error

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================
DATA_DIR="../RGBX_Semantic_Segmentation/data/images"
LABELS_DIR="../RGBX_Semantic_Segmentation/data/labels"
OUTPUT_DIR="results/modality_ablations"

# Training hyperparameters
BACKBONE="mit_b1"
EPOCHS=15
BATCH_SIZE=16
LR=0.02

# ============================================
# START EXPERIMENTS
# ============================================

echo "=============================================="
echo "Modality Ablation Experiments"
echo "=============================================="
echo "Data:         $DATA_DIR"
echo "Labels:       $LABELS_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Backbone:     $BACKBONE"
echo "Epochs:       $EPOCHS"
echo "Batch Size:   $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Architecture: Baseline FRM+FFM"
echo "=============================================="
echo ""
echo "Running 3 experiments in sequence:"
echo "  1. RGB + Thermal (no Event)"
echo "  2. RGB + Event (no Thermal)"
echo "  3. Thermal + Event (no RGB)"
echo ""
echo "Estimated time: 6-9 hours on A100"
echo "=============================================="
echo ""

# Run all experiments using the master script
python crossattentiondet/ablations/scripts/run_modality_ablations.py \
    --data "$DATA_DIR" \
    --labels "$LABELS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary: $OUTPUT_DIR/ablation_summary.json"
echo "=============================================="
