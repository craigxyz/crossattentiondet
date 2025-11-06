#!/bin/bash
#
# Convenience wrapper script to test all backbone variants
#
# Usage:
#   ./scripts/test_all_backbones.sh
#   ./scripts/test_all_backbones.sh data/ checkpoints/
#

# Default parameters
DATA_DIR="${1:-data}"
CHECKPOINT_DIR="${2:-checkpoints}"
BATCH_SIZE="${3:-2}"

echo "Testing all MiT backbone variants"
echo "Data: $DATA_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Batch size: $BATCH_SIZE"
echo ""

# Run the Python script
python scripts/test_all_backbones.py \
    --data "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size "$BATCH_SIZE"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ All backbones evaluated successfully!"
else
    echo ""
    echo "✗ Some backbones failed evaluation. Check the output above."
fi

exit $exit_code
