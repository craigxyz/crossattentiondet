#!/bin/bash
#
# Convenience wrapper script to train all backbone variants
#
# Usage:
#   ./scripts/train_all_backbones.sh
#   ./scripts/train_all_backbones.sh 10    # Train for 10 epochs
#

# Default parameters
DATA_DIR="${1:-data}"
EPOCHS="${2:-5}"
BATCH_SIZE="${3:-2}"
LR="${4:-0.005}"

echo "Training all MiT backbone variants"
echo "Data: $DATA_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo ""

# Run the Python script
python scripts/train_all_backbones.py \
    --data "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ All backbones trained successfully!"
else
    echo ""
    echo "✗ Some backbones failed to train. Check the output above."
fi

exit $exit_code
