#!/bin/bash
#
# Quick Start Script for Modality Ablation Experiments
#
# This script runs all modality ablation experiments with recommended settings
# for CVPR submission. Adjust paths and parameters as needed.
#

# Configuration
DATA_DIR="../RGBX_Semantic_Segmentation/data/images"
LABELS_DIR="../RGBX_Semantic_Segmentation/data/labels"
OUTPUT_DIR="results/modality_ablations"
BACKBONE="mit_b1"
EPOCHS=15
BATCH_SIZE=16
LR=0.02

# Print configuration
echo "=========================================="
echo "Modality Ablation Experiment Suite"
echo "=========================================="
echo "Data directory:   $DATA_DIR"
echo "Labels directory: $LABELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Backbone:         $BACKBONE"
echo "Epochs:           $EPOCHS"
echo "Batch size:       $BATCH_SIZE"
echo "Learning rate:    $LR"
echo "=========================================="
echo ""
echo "This will run 3 experiments:"
echo "  1. RGB + Thermal (no Event)"
echo "  2. RGB + Event (no Thermal)"
echo "  3. Thermal + Event (no RGB)"
echo ""
echo "NOTE: All modalities baseline already exists in other experiments"
echo ""
echo "Estimated time: 6-9 hours on A100 GPU (batch_size=32, 15 epochs)"
echo "=========================================="
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Run all experiments
python crossattentiondet/ablations/scripts/run_modality_ablations.py \
    --data "$DATA_DIR" \
    --labels "$LABELS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
