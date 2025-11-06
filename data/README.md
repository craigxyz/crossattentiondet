# Data Preparation Guide

This directory contains the dataset for CrossAttentionDet object detection.

## Directory Structure

```
data/
├── images/              # Multi-channel image data (.npy format)
│   ├── frame_001.npy
│   ├── frame_002.npy
│   └── ...
└── labels/              # YOLO-format annotations (.txt format)
    ├── frame_001.txt
    ├── frame_002.txt
    └── ...
```

## Data Format

#### Image Data (.npy files)

Images must be saved as NumPy arrays with shape `(H, W, 5)`:

```python
import numpy as np

# Example: Create 5-channel image
# Channel 0-2: RGB
# Channel 3: Thermal/IR
# Channel 4: Event
image = np.zeros((480, 640, 5), dtype=np.uint8)
image[:, :, 0:3] = rgb_image       # RGB channels
image[:, :, 3] = thermal_image     # Thermal channel
image[:, :, 4] = event_image       # Event channel

# Save as .npy
np.save('images/frame_001.npy', image)
```

**Channel Order:**
- Channels 0-2: RGB (Red, Green, Blue)
- Channel 3: Thermal/Infrared
- Channel 4: Event data

**Data Types:**
- Use `uint8` for 8-bit images (0-255)
- Use `uint16` for 16-bit thermal data if needed (will be normalized)

#### Label Data (YOLO format .txt files)

Each label file contains bounding box annotations in YOLO format:

```
class_id x_center y_center width height
```

Where:
- `class_id`: Integer class ID (0-indexed)
- `x_center`: Bounding box center X coordinate (normalized 0-1)
- `y_center`: Bounding box center Y coordinate (normalized 0-1)
- `width`: Bounding box width (normalized 0-1)
- `height`: Bounding box height (normalized 0-1)

**Example `labels/frame_001.txt`:**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.7 0.15 0.2
0 0.8 0.3 0.25 0.35
```

This defines 3 objects:
- Object 1: Class 0, centered at (50%, 50%), size 30% × 40%
- Object 2: Class 1, centered at (20%, 70%), size 15% × 20%
- Object 3: Class 0, centered at (80%, 30%), size 25% × 35%

## Data Preparation Scripts

### Converting Separate Modalities to 5-Channel Format

If you have separate RGB, thermal, and event images:

```python
import numpy as np
import cv2
from pathlib import Path

def combine_modalities(rgb_path, thermal_path, event_path, output_path):
    """
    Combine RGB, thermal, and event images into 5-channel .npy format.

    Args:
        rgb_path: Path to RGB image (3-channel)
        thermal_path: Path to thermal image (1-channel grayscale)
        event_path: Path to event image (1-channel grayscale)
        output_path: Output path for .npy file
    """
    # Load images
    rgb = cv2.imread(str(rgb_path))  # Shape: (H, W, 3)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    thermal = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)  # Shape: (H, W)
    event = cv2.imread(str(event_path), cv2.IMREAD_GRAYSCALE)  # Shape: (H, W)

    # Ensure all images have same dimensions
    h, w = rgb.shape[:2]
    thermal = cv2.resize(thermal, (w, h))
    event = cv2.resize(event, (w, h))

    # Combine into 5-channel array
    combined = np.zeros((h, w, 5), dtype=np.uint8)
    combined[:, :, 0:3] = rgb
    combined[:, :, 3] = thermal
    combined[:, :, 4] = event

    # Save
    np.save(output_path, combined)
    print(f"Saved {output_path}")

# Example usage
rgb_dir = Path("raw_data/rgb")
thermal_dir = Path("raw_data/thermal")
event_dir = Path("raw_data/event")
output_dir = Path("data/images")
output_dir.mkdir(parents=True, exist_ok=True)

for rgb_file in rgb_dir.glob("*.png"):
    frame_name = rgb_file.stem
    thermal_file = thermal_dir / f"{frame_name}.png"
    event_file = event_dir / f"{frame_name}.png"

    if thermal_file.exists() and event_file.exists():
        output_file = output_dir / f"{frame_name}.npy"
        combine_modalities(rgb_file, thermal_file, event_file, output_file)
```

### Converting COCO/Pascal VOC to YOLO Format

```python
def coco_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bounding box (x_min, y_min, width, height) to YOLO format.

    Args:
        bbox: [x_min, y_min, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        [x_center, y_center, width, height] normalized to 0-1
    """
    x_min, y_min, box_w, box_h = bbox

    x_center = (x_min + box_w / 2) / img_width
    y_center = (y_min + box_h / 2) / img_height
    width = box_w / img_width
    height = box_h / img_height

    return [x_center, y_center, width, height]

# Example usage
# For each image annotation:
img_width, img_height = 640, 480
coco_bbox = [100, 150, 200, 180]  # x_min, y_min, width, height
yolo_bbox = coco_to_yolo(coco_bbox, img_width, img_height)
class_id = 0

# Write to label file
with open('labels/frame_001.txt', 'w') as f:
    f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
```

## Data Validation

### Verify 5-Channel Data

```python
import numpy as np
import matplotlib.pyplot as plt

# Load and inspect
data = np.load('images/frame_001.npy')
print(f"Shape: {data.shape}")  # Should be (H, W, 5)
print(f"Dtype: {data.dtype}")  # Should be uint8 or uint16

# Visualize each channel
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
channel_names = ['Red', 'Green', 'Blue', 'Thermal', 'Event']

for i, (ax, name) in enumerate(zip(axes, channel_names)):
    ax.imshow(data[:, :, i], cmap='gray' if i >= 3 else None)
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.savefig('data_visualization.png')
```

### Verify YOLO Labels

```python
import numpy as np
import cv2

def visualize_yolo_labels(image_path, label_path):
    """Visualize YOLO format labels on image."""
    # Load image
    img = np.load(image_path)
    rgb = img[:, :, 0:3]  # Extract RGB channels
    h, w = rgb.shape[:2]

    # Load labels
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_c, y_c, box_w, box_h = map(float, line.strip().split())

            # Convert YOLO to pixel coordinates
            x_min = int((x_c - box_w/2) * w)
            y_min = int((y_c - box_h/2) * h)
            x_max = int((x_c + box_w/2) * w)
            y_max = int((y_c + box_h/2) * h)

            # Draw bounding box
            cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(rgb, f"Class {int(class_id)}", (x_min, y_min-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save visualization
    cv2.imwrite('label_visualization.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# Example usage
visualize_yolo_labels('images/frame_001.npy', 'labels/frame_001.txt')
```

## Common Issues

### Issue: Mismatched image dimensions
**Solution**: Ensure all modalities (RGB, thermal, event) are resized to the same dimensions before combining.

### Issue: Incorrect normalization
**Solution**: YOLO coordinates must be normalized to [0, 1]. Check that your conversion script divides by image width/height.

### Issue: Missing label files
**Solution**: Ensure each .npy image has a corresponding .txt label file with the same name.

### Issue: Class ID mismatch
**Solution**: Verify class IDs start from 0 and match your model's num_classes configuration.

## Dataset Statistics

After preparing your dataset, calculate statistics:

```python
import numpy as np
from pathlib import Path

# Count samples
image_files = list(Path('images').glob('*.npy'))
label_files = list(Path('labels').glob('*.txt'))

print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(label_files)}")

# Count objects per class
class_counts = {}
for label_file in label_files:
    with open(label_file) as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

print("\nObjects per class:")
for class_id, count in sorted(class_counts.items()):
    print(f"  Class {class_id}: {count} objects")
```

## Next Steps

1. Prepare your raw data (RGB, thermal, event images)
2. Convert to 5-channel .npy format
3. Create YOLO format labels
4. Validate data with visualization scripts
5. Update class count in scripts/train.py or ablation configs
6. Begin training!
