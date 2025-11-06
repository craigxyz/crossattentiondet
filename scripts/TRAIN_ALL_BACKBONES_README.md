# Train All Backbones Script

## Overview

The `train_all_backbones.py` script trains all MiT backbone variants sequentially, making it easy to compare performance across different model sizes.

## Features

- **Serial Training**: Trains each backbone variant one after another
- **Separate Checkpoints**: Saves a unique checkpoint for each variant
- **Timing Comparison**: Tracks and compares training time for each backbone
- **Flexible Selection**: Train all backbones or select specific ones
- **Summary Report**: Generates a detailed summary with timing statistics

## Usage

### Basic Usage

```bash
# Train all backbones for 5 epochs (default)
python scripts/train_all_backbones.py --data data/ --epochs 5
```

### Advanced Options

```bash
# Custom epochs and batch size
python scripts/train_all_backbones.py \
    --data data/ \
    --epochs 10 \
    --batch-size 4 \
    --lr 0.005

# Train only specific backbones
python scripts/train_all_backbones.py \
    --data data/ \
    --epochs 5 \
    --backbones mit_b0 mit_b1 mit_b5

# Specify checkpoint directory
python scripts/train_all_backbones.py \
    --data data/ \
    --epochs 5 \
    --checkpoint-dir my_experiments/run1
```

### Using the Bash Wrapper

```bash
# Syntax: ./train_all_backbones.sh [data_dir] [epochs] [batch_size] [lr]

# Default settings (data/, 5 epochs, batch_size 2, lr 0.005)
./scripts/train_all_backbones.sh

# Custom settings
./scripts/train_all_backbones.sh data/ 10 4 0.005
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | **required** | Path to data directory with images/ and labels/ |
| `--epochs` | int | 5 | Number of training epochs per backbone |
| `--batch-size` | int | 2 | Batch size for training |
| `--lr` | float | 0.005 | Learning rate |
| `--backbones` | list | all | List of backbones to train |
| `--checkpoint-dir` | str | checkpoints | Directory to save model checkpoints |

## Output

### Console Output

The script provides detailed progress information:

```
================================================================================
TRAINING ALL BACKBONE VARIANTS
================================================================================
Data directory: data/
Epochs per backbone: 5
Batch size: 2
Learning rate: 0.005
Backbones to train: mit_b0, mit_b1, mit_b2, mit_b4, mit_b5
Checkpoint directory: checkpoints
================================================================================

################################################################################
# TRAINING BACKBONE 1/5: MIT_B0
################################################################################

Using backbone: mit_b0

1. Setting up dataset...
2. Initializing DataLoader...
3. Building Multi-modal FPN backbone...
...

MIT_B0 training completed!
Time elapsed: 245.32 seconds (4.09 minutes)
Model saved to: checkpoints/crossattentiondet_mit_b0.pth

Progress: 1/5 backbones processed (✓ 1 successful, ✗ 0 failed)

...

================================================================================
TRAINING SUMMARY
================================================================================

Total time: 1523.45 seconds (25.39 minutes)
Epochs per backbone: 5

Results:
--------------------------------------------------------------------------------
  ✓ mit_b0   |   4.09 min | checkpoints/crossattentiondet_mit_b0.pth
  ✓ mit_b1   |   6.23 min | checkpoints/crossattentiondet_mit_b1.pth
  ✓ mit_b2   |   8.45 min | checkpoints/crossattentiondet_mit_b2.pth
  ✓ mit_b4   |  12.67 min | checkpoints/crossattentiondet_mit_b4.pth
  ✓ mit_b5   |  15.12 min | checkpoints/crossattentiondet_mit_b5.pth
--------------------------------------------------------------------------------

Successful: 5/5
Failed: 0/5
Average training time (successful): 9.31 minutes

Speed comparison (successful backbones):
  mit_b0  :   4.09 min
  mit_b1  :   6.23 min
  mit_b2  :   8.45 min
  mit_b4  :  12.67 min
  mit_b5  :  15.12 min

Summary saved to: checkpoints/training_summary_20250115_143052.txt
================================================================================
ALL TRAINING COMPLETED
================================================================================
```

### Generated Files

The script creates:

1. **Model Checkpoints**: One per backbone variant
   ```
   checkpoints/
   ├── crossattentiondet_mit_b0.pth
   ├── crossattentiondet_mit_b1.pth
   ├── crossattentiondet_mit_b2.pth
   ├── crossattentiondet_mit_b4.pth
   ├── crossattentiondet_mit_b5.pth
   └── training_summary_20250115_143052.txt
   ```

2. **Summary Report**: Timestamped text file with complete training statistics

## Example Workflows

### Rapid Prototyping

Train only the smallest/fastest backbones for quick iteration:

```bash
python scripts/train_all_backbones.py \
    --data data/ \
    --epochs 3 \
    --backbones mit_b0 mit_b1
```

### Full Comparison

Train all backbones for comprehensive evaluation:

```bash
python scripts/train_all_backbones.py \
    --data data/ \
    --epochs 15 \
    --batch-size 4 \
    --checkpoint-dir experiments/full_comparison
```

### Production Models

Train only the larger, more accurate backbones:

```bash
python scripts/train_all_backbones.py \
    --data data/ \
    --epochs 20 \
    --backbones mit_b4 mit_b5 \
    --checkpoint-dir production_models
```

## Tips

1. **Start Small**: Try 3-5 epochs first to verify everything works before committing to longer training runs

2. **Monitor Resources**: Larger backbones (mit_b4, mit_b5) require more GPU memory. Consider reducing batch size if you encounter OOM errors.

3. **Save Summaries**: The timestamped summary files make it easy to compare multiple experimental runs.

4. **Checkpoint Management**: Use `--checkpoint-dir` to organize different experiments:
   ```bash
   experiments/
   ├── run1_baseline/
   ├── run2_augmented/
   └── run3_tuned/
   ```

5. **Resume Training**: If training fails partway through, you can resume by specifying only the remaining backbones with `--backbones`.

## Troubleshooting

**Out of Memory Error**:
```bash
# Reduce batch size
python scripts/train_all_backbones.py --data data/ --epochs 5 --batch-size 1
```

**Specific Backbone Failing**:
```bash
# Skip problematic backbone and train others
python scripts/train_all_backbones.py \
    --data data/ \
    --epochs 5 \
    --backbones mit_b0 mit_b1 mit_b2  # Skip mit_b4 and mit_b5
```

**Long Training Times**:
```bash
# Train fewer epochs for initial experiments
python scripts/train_all_backbones.py --data data/ --epochs 2
```

## Exit Codes

- `0`: All backbones trained successfully
- `1`: One or more backbones failed to train (details in output)
