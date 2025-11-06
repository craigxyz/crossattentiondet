# CrossAttentionDet

Cross-attention multi-modal object detection using RGB, thermal, and event camera data.

## Overview

CrossAttentionDet is a multi-modal object detection framework that processes RGB, thermal, and event camera inputs through cross-attention fusion mechanisms. The architecture combines hierarchical feature extraction with cross-modal attention to improve detection performance across varying environmental conditions.

## Architecture

The framework consists of three main components:

**Multi-Modal Encoder**
- Dual-stream transformer architecture based on SegFormer MiT (Mix Transformer)
- Configurable backbone variants (mit_b0 through mit_b5) for different speed/accuracy trade-offs
- Separate pathways for RGB and auxiliary modalities (thermal, event)
- Hierarchical feature extraction at multiple scales

**Cross-Modal Fusion**
- Feature Fusion Module (FFM) with cross-attention mechanisms
- Feature Rectify Module (FRM) for adaptive modality weighting
- Channel and spatial attention for multi-modal integration

**Detection Head**
- Feature Pyramid Network (FPN) for multi-scale features
- Region Proposal Network (RPN) for object proposals
- Faster R-CNN detection head for classification and localization

## Installation

Requirements:
- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- CUDA 10.2 or higher (for GPU training)

Setup:
```bash
git clone https://github.com/craigxyz/crossattentiondet.git
cd crossattentiondet
pip install -r requirements.txt
```

## Data Format

### Input Images

Images must be 5-channel NumPy arrays saved as .npy files with shape (H, W, 5):
- Channels 0-2: RGB
- Channel 3: Thermal or infrared
- Channel 4: Event camera data

Example:
```python
import numpy as np

image = np.zeros((480, 640, 5), dtype=np.uint8)
image[:, :, 0:3] = rgb_data
image[:, :, 3] = thermal_data
image[:, :, 4] = event_data

np.save('data/images/frame_001.npy', image)
```

### Annotations

Labels use YOLO format (one line per object):
```
class_id x_center y_center width height
```

All coordinates are normalized to [0, 1].

Example (data/labels/frame_001.txt):
```
0 0.5 0.5 0.3 0.4
1 0.2 0.7 0.15 0.2
```

### Directory Structure

Users must create the following directories before training:
```
data/
├── images/    (place .npy files here)
└── labels/    (place .txt files here)
```

## Usage

### Training

```bash
python scripts/train.py --data data/ --epochs 15 --batch-size 2
```

Optional arguments:
- `--backbone`: Backbone variant (default: mit_b1)
  - `mit_b0`: Smallest, fastest (32M parameters)
  - `mit_b1`: Balanced, default (13.5M parameters)
  - `mit_b2`: Base model (24.7M parameters)
  - `mit_b4`: Large, high accuracy (61.4M parameters)
  - `mit_b5`: Largest, best accuracy (81.9M parameters)
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size (default: 2)
- `--lr`: Learning rate (default: 0.005)
- `--model`: Path to save model checkpoint (default: crossattentiondet.pth)

### Backbone Selection

You can select different encoder backbones to balance speed vs. accuracy:

```bash
# Fastest training/inference (recommended for prototyping)
python scripts/train.py --data data/ --epochs 15 --backbone mit_b0

# Default balanced configuration
python scripts/train.py --data data/ --epochs 15 --backbone mit_b1

# Higher accuracy for production
python scripts/train.py --data data/ --epochs 15 --backbone mit_b5
```

**Backbone Comparison:**

| Variant | Parameters | Depths | Embed Dims | Use Case |
|---------|-----------|---------|------------|----------|
| mit_b0 | ~3.7M | [2,2,2,2] | [32,64,160,256] | Fast prototyping, edge devices |
| mit_b1 | ~13.5M | [2,2,2,2] | [64,128,320,512] | Default, balanced |
| mit_b2 | ~24.7M | [3,4,6,3] | [64,128,320,512] | Higher accuracy |
| mit_b4 | ~61.4M | [3,8,27,3] | [64,128,320,512] | Large-scale detection |
| mit_b5 | ~81.9M | [3,6,40,3] | [64,128,320,512] | Maximum accuracy |

### Training All Backbones for Comparison

To train all backbone variants serially for comparison:

```bash
# Train all backbones for 5 epochs each (default)
python scripts/train_all_backbones.py --data data/ --epochs 5

# Train all backbones for 10 epochs with custom batch size
python scripts/train_all_backbones.py --data data/ --epochs 10 --batch-size 4 --lr 0.005

# Train only specific backbones
python scripts/train_all_backbones.py --data data/ --epochs 5 --backbones mit_b0 mit_b1 mit_b2

# Using the convenience bash wrapper
./scripts/train_all_backbones.sh data/ 5 2 0.005
```

This script will:
- Train each backbone variant sequentially
- Save separate checkpoints for each variant (e.g., `checkpoints/crossattentiondet_mit_b0.pth`)
- Track training time for each variant
- Generate a summary report with timing comparisons
- Create a timestamped summary file in the checkpoints directory

### Evaluation

```bash
python scripts/test.py --data data/ --model crossattentiondet.pth
```

Results are saved to the test_results/ directory with visualizations of ground truth and predicted bounding boxes.

### Visualization

```bash
python scripts/visualize.py --vis 0
```

Visualizes the sample at index 0 from the dataset.

## Configuration

The Config class in crossattentiondet/config.py contains all hyperparameters:

```python
num_epochs = 15              # Training epochs
batch_size = 2               # Batch size
learning_rate = 0.005        # Learning rate
fpn_out_channels = 256       # FPN output channels
```

Modify these parameters based on your hardware and dataset requirements.

## Technical Details

### Cross-Modal Fusion

The Feature Fusion Module (FFM) combines features from RGB and auxiliary modality streams using cross-attention. Query vectors from one modality attend to key-value pairs from the other modality, enabling bidirectional information flow.

The Feature Rectify Module (FRM) adaptively weights contributions from each modality using both channel and spatial attention mechanisms. This improves robustness when one modality contains noise or missing data.

### Training

- Optimizer: SGD with momentum
- Loss: Combined classification and bounding box regression from Faster R-CNN
- Data split: Automatic 80/20 train/test split
- Device: Automatic GPU detection with CPU fallback

## Project Structure

```
crossattentiondet/
├── crossattentiondet/        # Main package
│   ├── models/              # Model architecture
│   ├── data/                # Dataset and transforms
│   ├── training/            # Training and evaluation
│   ├── utils/               # Utilities and visualization
│   └── config.py            # Configuration
├── scripts/                 # Entry point scripts
│   ├── train.py            # Training script
│   ├── test.py             # Evaluation script
│   └── visualize.py        # Visualization script
├── requirements.txt
└── README.md
```

## License

MIT License
