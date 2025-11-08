#!/bin/bash
# Re-run all CSSA ablations with consistent aggressive hyperparameters
# This ensures a fair architectural comparison

DATA_DIR="../RGBX_Semantic_Segmentation/data/images"
LABELS_DIR="../RGBX_Semantic_Segmentation/data/labels"
OUTPUT_BASE="results/cssa_ablations_v2"
BACKBONE="mit_b1"
EPOCHS=15
BATCH_SIZE=16
LR=0.02
THRESH=0.5
KERNEL=3

# Common arguments
COMMON_ARGS="--data $DATA_DIR --labels $LABELS_DIR --backbone $BACKBONE --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --cssa-thresh $THRESH --cssa-kernel $KERNEL"

echo "========================================================================"
echo "       CSSA Ablation Study v2 - Consistent Hyperparameters"
echo "========================================================================"
echo "Backbone: $BACKBONE"
echo "Epochs: $EPOCHS | Batch Size: $BATCH_SIZE | Learning Rate: $LR"
echo "Threshold: $THRESH | Kernel: $KERNEL"
echo ""
echo "All experiments will use IDENTICAL training configuration"
echo "Expected time per experiment: ~3.5-4 hours"
echo "Total expected time: ~24-28 hours"
echo "========================================================================"
echo ""

# Experiment 1: Stage 1
echo "[1/7] Running exp_001_s1_t0.5 (Stage 1 only)..."
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_001_s1_t0.5 \
    --cssa-stages 1

# Experiment 2: Stage 2
echo "[2/7] Running exp_002_s2_t0.5 (Stage 2 only)..."
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_002_s2_t0.5 \
    --cssa-stages 2

# Experiment 3: Stage 3
echo "[3/7] Running exp_003_s3_t0.5 (Stage 3 only)..."
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_003_s3_t0.5 \
    --cssa-stages 3

# Experiment 4: Stage 4
echo "[4/7] Running exp_004_s4_t0.5 (Stage 4 only)..."
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_004_s4_t0.5 \
    --cssa-stages 4

# Experiment 5: Stages 2+3
echo "[5/7] Running exp_005_s23_t0.5 (Stages 2+3)..."
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_005_s23_t0.5 \
    --cssa-stages 2,3

# Experiment 6: Stages 3+4
echo "[6/7] Running exp_006_s34_t0.5 (Stages 3+4)..."
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_006_s34_t0.5 \
    --cssa-stages 3,4

# Experiment 7: All stages
echo "[7/7] Running exp_007_s1234_t0.5 (All stages)..."
python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_007_s1234_t0.5 \
    --cssa-stages 1,2,3,4

echo ""
echo "========================================================================"
echo "                    ALL EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "Next steps:"
echo "1. Run analysis: python crossattentiondet/ablations/scripts/analyze_cssa_results.py"
echo "2. Compare with v1 results in: results/cssa_ablations/"
echo "========================================================================"
