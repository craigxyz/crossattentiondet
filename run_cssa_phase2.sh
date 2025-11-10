#!/bin/bash
################################################################################
# CSSA Phase 2: Minimum Viable Experiments (5 experiments)
#
# Purpose: Validate Phase 1 findings and test threshold sensitivity
#
# All experiments use aggressive training:
#   - 15 epochs, batch_size=16, lr=0.02, kernel=3
#
# Estimated duration: ~19 hours (5 experiments × ~3.8 hours each)
################################################################################

DATA_DIR="../RGBX_Semantic_Segmentation/data/images"
LABELS_DIR="../RGBX_Semantic_Segmentation/data/labels"
OUTPUT_BASE="results/cssa_ablations_phase2"
BACKBONE="mit_b1"
EPOCHS=15
BATCH_SIZE=16
LR=0.02
KERNEL=3

# Common arguments
COMMON_ARGS="--data $DATA_DIR --labels $LABELS_DIR --backbone $BACKBONE --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --cssa-kernel $KERNEL"

# Create output directory
mkdir -p $OUTPUT_BASE

# Log file
LOG_FILE="$OUTPUT_BASE/phase2_master_log.txt"

################################################################################
# Helper function for logging
################################################################################
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

################################################################################
# Start Phase 2
################################################################################
log "========================================================================"
log "                   CSSA PHASE 2: THRESHOLD ABLATION                    "
log "========================================================================"
log "Configuration: 15 epochs, batch_size=16, lr=0.02, kernel=3"
log "Total experiments: 5"
log "Estimated duration: ~19 hours"
log "Output directory: $OUTPUT_BASE"
log ""

START_TIME=$(date +%s)

################################################################################
# Experiment 1: Stage 1, threshold=0.3 (aggressive channel switching)
################################################################################
log "========================================================================"
log "           [1/5] exp_008_s1_t0.3 - Stage 1, threshold=0.3              "
log "========================================================================"
log "Testing aggressive channel switching on best configuration (Stage 1)"
log ""

python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_008_s1_t0.3 \
    --cssa-stages 1 \
    --cssa-thresh 0.3

if [ $? -eq 0 ]; then
    log "✓ exp_008_s1_t0.3 completed successfully"
else
    log "✗ exp_008_s1_t0.3 FAILED"
    exit 1
fi
log ""

################################################################################
# Experiment 2: Stage 1, threshold=0.7 (conservative channel switching)
################################################################################
log "========================================================================"
log "           [2/5] exp_009_s1_t0.7 - Stage 1, threshold=0.7              "
log "========================================================================"
log "Testing conservative channel switching on best configuration (Stage 1)"
log ""

python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_009_s1_t0.7 \
    --cssa-stages 1 \
    --cssa-thresh 0.7

if [ $? -eq 0 ]; then
    log "✓ exp_009_s1_t0.7 completed successfully"
else
    log "✗ exp_009_s1_t0.7 FAILED"
    exit 1
fi
log ""

################################################################################
# Experiment 3: Stage 4, threshold=0.3 (validate runner-up)
################################################################################
log "========================================================================"
log "           [3/5] exp_010_s4_t0.3 - Stage 4, threshold=0.3              "
log "========================================================================"
log "Testing if Stage 4 (runner-up) can beat Stage 1 with different threshold"
log ""

python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_010_s4_t0.3 \
    --cssa-stages 4 \
    --cssa-thresh 0.3

if [ $? -eq 0 ]; then
    log "✓ exp_010_s4_t0.3 completed successfully"
else
    log "✗ exp_010_s4_t0.3 FAILED"
    exit 1
fi
log ""

################################################################################
# Experiment 4: Stage 3, threshold=0.3 (originally predicted best)
################################################################################
log "========================================================================"
log "           [4/5] exp_011_s3_t0.3 - Stage 3, threshold=0.3              "
log "========================================================================"
log "Testing Stage 3 with different threshold (originally predicted optimal)"
log ""

python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_011_s3_t0.3 \
    --cssa-stages 3 \
    --cssa-thresh 0.3

if [ $? -eq 0 ]; then
    log "✓ exp_011_s3_t0.3 completed successfully"
else
    log "✗ exp_011_s3_t0.3 FAILED"
    exit 1
fi
log ""

################################################################################
# Experiment 5: Stages 2+3, threshold=0.3 (anomaly check)
################################################################################
log "========================================================================"
log "          [5/5] exp_012_s23_t0.3 - Stages 2+3, threshold=0.3           "
log "========================================================================"
log "Testing multi-stage config with exceptional large object performance"
log "(Phase 1 showed 86.24% mAP_large - checking if threshold-dependent)"
log ""

python -u crossattentiondet/ablations/scripts/train_cssa_ablation.py \
    $COMMON_ARGS \
    --output-dir $OUTPUT_BASE/exp_012_s23_t0.3 \
    --cssa-stages 2,3 \
    --cssa-thresh 0.3

if [ $? -eq 0 ]; then
    log "✓ exp_012_s23_t0.3 completed successfully"
else
    log "✗ exp_012_s23_t0.3 FAILED"
    exit 1
fi
log ""

################################################################################
# Summary
################################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

log "========================================================================"
log "                    PHASE 2 EXPERIMENTS COMPLETE!                      "
log "========================================================================"
log "Completed: 5/5 experiments"
log "Total time: ${HOURS}h ${MINUTES}m"
log "Results saved to: $OUTPUT_BASE/"
log ""
log "Experiment summary:"
log "  [1] exp_008_s1_t0.3   - Stage 1, threshold=0.3"
log "  [2] exp_009_s1_t0.7   - Stage 1, threshold=0.7"
log "  [3] exp_010_s4_t0.3   - Stage 4, threshold=0.3"
log "  [4] exp_011_s3_t0.3   - Stage 3, threshold=0.3"
log "  [5] exp_012_s23_t0.3  - Stages 2+3, threshold=0.3"
log ""
log "Next steps:"
log "  1. Review results in: $OUTPUT_BASE/*/final_results.json"
log "  2. Run analysis: python crossattentiondet/ablations/scripts/analyze_cssa_results.py"
log "  3. Compare with Phase 1: results/cssa_ablations/"
log ""
log "Research questions answered:"
log "  ✓ Is Stage 1 threshold-sensitive? (compare exp_001_fast, exp_008, exp_009)"
log "  ✓ Can other stages beat Stage 1 with optimal threshold?"
log "  ✓ Does multi-stage exceptional large object performance persist?"
log "========================================================================"
