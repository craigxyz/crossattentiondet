# Dataset & Modalities

**RGBX Multi-Modal Object Detection Dataset**

[← Back to Index](00_INDEX.md) | [← Previous: Architecture](02_ARCHITECTURE_DEEP_DIVE.md) | [Next: Ablation Studies →](04_ABLATION_STUDIES.md)

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Format & Structure](#data-format--structure)
3. [Modality Specifications](#modality-specifications)
4. [Data Loading Pipeline](#data-loading-pipeline)
5. [Train/Test Splits](#traintest-splits)
6. [Data Augmentation](#data-augmentation)
7. [Dataset Statistics](#dataset-statistics)

---

## Dataset Overview

### RGBX Multi-Modal Dataset

**Location:** `/mmfs1/project/pva23/csi3/RGBX_Semantic_Segmentation/data/`

**Key Statistics:**
- **Total Images:** 10,489 .npy files
- **Valid Annotations:** 9,750 images (93% coverage)
- **Total Bounding Boxes:** ~24,223
- **Average Objects per Image:** ~3.14
- **Object Classes:** 1 primary class (+ background for detection)
- **Modalities:** 3 (RGB, Thermal, Event)
- **Channels:** 5 total (RGB=3, Thermal=1, Event=1)

### Directory Structure

```
../RGBX_Semantic_Segmentation/data/
├── images/                    # 10,489 .npy files
│   ├── frame_000001.npy      # (H, W, 5) arrays
│   ├── frame_000002.npy
│   └── ...
├── labels/                    # 9,750 .txt files (YOLO format)
│   ├── frame_000001.txt      # class x_center y_center width height
│   ├── frame_000002.txt
│   └── ...
├── daytest1/                  # Day-time test set (separate)
└── nighttest1/                # Night-time test set (separate)
```

**Test Sets:**
- `daytest1/`: Evaluates RGB performance (good lighting)
- `nighttest1/`: Evaluates Thermal/Event performance (low/no light)

---

## Data Format & Structure

### Image Format

**File Type:** NumPy arrays (.npy)

**Shape:** `(H, W, 5)` where H and W vary by image

**Channel Layout:**
```python
image = np.load('frame_000001.npy')  # Shape: (H, W, 5)

# Channel 0-2: RGB
rgb = image[:, :, 0:3]      # (H, W, 3), uint8, range [0, 255]
  red   = image[:, :, 0]
  green = image[:, :, 1]
  blue  = image[:, :, 2]

# Channel 3: Thermal/Infrared
thermal = image[:, :, 3]    # (H, W), uint8, range [0, 255]

# Channel 4: Event Camera
event = image[:, :, 4]      # (H, W), uint8, range [0, 255]
```

**Typical Resolutions:**
- Most common: 301×391, 304×400, 320×240
- Range: ~200×200 to ~640×480
- Variable aspect ratios (not fixed size)

**Data Type:** uint8 (all channels)

**Storage:** ~6.5 GB total for all images

### Label Format

**File Type:** Text files (.txt), YOLO format

**Structure:** One line per object
```
<class_id> <x_center> <y_center> <width> <height>
```

**Coordinate System:**
- All values normalized to [0, 1]
- `x_center`, `y_center`: Box center relative to image width/height
- `width`, `height`: Box dimensions relative to image width/height
- `class_id`: 0-indexed in files, converted to 1-indexed for Faster R-CNN (background=0)

**Example label file (`frame_000123.txt`):**
```
0 0.5234 0.4678 0.2341 0.1897
0 0.7821 0.3234 0.1456 0.1567
0 0.2145 0.8123 0.0987 0.0654
```
→ 3 objects, all class 0

**Conversion to COCO Format:**
```python
# YOLO → COCO conversion
x_center, y_center, w_norm, h_norm = yolo_bbox  # Normalized [0, 1]

# Convert to absolute pixel coordinates
x_center_abs = x_center * image_width
y_center_abs = y_center * image_height
w_abs = w_norm * image_width
h_abs = h_norm * image_height

# Convert to COCO format (x_min, y_min, x_max, y_max)
x_min = x_center_abs - w_abs / 2
y_min = y_center_abs - h_abs / 2
x_max = x_center_abs + w_abs / 2
y_max = y_center_abs + h_abs / 2

coco_bbox = [x_min, y_min, x_max, y_max]
```

**Storage:** ~5.8 MB total for all labels

---

## Modality Specifications

### RGB Channels (0-2)

**Sensor Type:** Standard RGB camera

**Characteristics:**
- **Strengths:**
  - High spatial resolution
  - Rich color/texture information
  - Good for shape, appearance-based detection
- **Weaknesses:**
  - Fails in low-light conditions
  - Poor performance at night
  - Sensitive to lighting changes, shadows

**Typical Use Cases:**
- Daytime object detection
- Well-lit environments
- Color-based object recognition

**Normalization (ImageNet stats):**
```python
rgb_mean = [0.485, 0.456, 0.406]  # R, G, B
rgb_std  = [0.229, 0.224, 0.225]
```

### Thermal/Infrared Channel (3)

**Sensor Type:** Thermal infrared camera (LWIR: Long-Wave Infrared, 8-14 μm)

**Characteristics:**
- **Strengths:**
  - Lighting-invariant (works day and night)
  - Detects heat signatures (humans, vehicles, animals)
  - Robust to shadows, fog, smoke
- **Weaknesses:**
  - Lower spatial resolution than RGB
  - Less texture/color information
  - Thermal contrast may be low for cold objects

**Typical Use Cases:**
- Night-time object detection
- Low-visibility conditions (fog, rain, darkness)
- Detecting warm-blooded objects (humans, animals)

**Normalization:**
```python
thermal_mean = 0.5
thermal_std  = 0.5
```

**Physical Interpretation:**
- Brighter pixels: Warmer regions (high IR radiation)
- Darker pixels: Cooler regions (low IR radiation)

### Event Camera Channel (4)

**Sensor Type:** Event-based camera (Dynamic Vision Sensor, DVS)

**Characteristics:**
- **Strengths:**
  - Ultra-high temporal resolution (microseconds)
  - Low latency
  - High dynamic range (>120 dB vs. ~60 dB for RGB)
  - Asynchronous, sparse output (only changes recorded)
- **Weaknesses:**
  - Requires motion to generate events
  - Sparse data (static scenes produce no events)
  - Less intuitive than frame-based cameras

**Typical Use Cases:**
- High-speed motion detection
- Dynamic scene understanding
- Lighting-invariant temporal information

**Normalization:**
```python
event_mean = 0.5
event_std  = 0.5
```

**Physical Interpretation:**
- Events triggered by pixel brightness changes
- Accumulated events represented as intensity in frame format
- Brighter: More events (more temporal changes)

### Modality Complementarity

**Why RGB + Thermal + Event?**

| Condition | RGB | Thermal | Event | Best Modality |
|-----------|-----|---------|-------|---------------|
| Daytime, static | ✅ Excellent | ⚠️ OK | ❌ Poor | **RGB** |
| Daytime, dynamic | ✅ Good | ⚠️ OK | ✅ Good | **RGB + Event** |
| Night, static | ❌ Poor | ✅ Excellent | ❌ Poor | **Thermal** |
| Night, dynamic | ❌ Poor | ✅ Excellent | ✅ Good | **Thermal + Event** |
| Low-light, motion | ⚠️ OK | ✅ Good | ✅ Excellent | **Thermal + Event** |
| Fog/Smoke | ❌ Poor | ✅ Good | ⚠️ OK | **Thermal** |

**Hypothesis:** Fusing all three modalities provides robustness across diverse environmental conditions, outperforming any single modality.

---

## Data Loading Pipeline

**File:** `crossattentiondet/data/dataset.py` (126 lines)

### NpyYoloDataset Class

```python
class NpyYoloDataset(Dataset):
    """
    Dataset for loading .npy images and YOLO format labels.

    Args:
        image_dir (str): Path to .npy image directory
        label_dir (str): Path to .txt label directory
        mode (str): 'train' or 'test'
        test_size (float): Test set proportion (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
    """

    def __init__(self, image_dir, label_dir, mode='train',
                 test_size=0.2, random_state=42):
        # Find all .npy files
        all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])

        # Find valid files (with non-empty labels)
        valid_pairs = []
        for img_file in all_images:
            label_file = img_file.replace('.npy', '.txt')
            label_path = os.path.join(label_dir, label_file)

            if os.path.exists(label_path):
                # Check if label file is non-empty
                with open(label_path, 'r') as f:
                    if f.read().strip():  # Non-empty
                        valid_pairs.append(img_file)

        # Train/test split
        train_files, test_files = train_test_split(
            valid_pairs,
            test_size=test_size,
            random_state=random_state
        )

        self.files = train_files if mode == 'train' else test_files
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            image: (5, H, W) tensor
            target: dict with 'boxes' (N, 4) and 'labels' (N,)
        """
        # Load image
        img_file = self.files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = np.load(img_path)  # (H, W, 5)

        # Convert to tensor and permute
        image = torch.from_numpy(image).float()  # (H, W, 5)
        image = image.permute(2, 0, 1)  # (5, H, W)

        # Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5]).view(5, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225, 0.5, 0.5]).view(5, 1, 1)
        image = (image / 255.0 - image_mean) / image_std

        # Load labels
        label_file = img_file.replace('.npy', '.txt')
        label_path = os.path.join(self.label_dir, label_file)

        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_c, y_c, w, h = map(float, line.strip().split())

                # YOLO → COCO conversion
                _, H, W = image.shape
                x_min = (x_c - w / 2) * W
                y_min = (y_c - h / 2) * H
                x_max = (x_c + w / 2) * W
                y_max = (y_c + h / 2) * H

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id) + 1)  # 0-indexed → 1-indexed (0 is background)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        return image, target
```

### Custom Collate Function

**Purpose:** Handle variable image sizes and number of objects per image.

```python
def collate_fn(batch):
    """
    Batch images and targets (variable sizes).

    Args:
        batch: List of (image, target) tuples

    Returns:
        images: List of (5, H, W) tensors (variable H, W)
        targets: List of dicts with 'boxes' and 'labels'
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets  # Note: Not stacked (variable sizes)
```

**DataLoader Usage:**
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)
```

---

## Train/Test Splits

### Split Configuration

**File:** `crossattentiondet/config.py:18-19`

```python
test_size = 0.2      # 20% test, 80% train
random_state = 42    # Fixed seed for reproducibility
```

### Split Statistics

| Split | Images | Percentage | Boxes (approx) |
|-------|--------|------------|----------------|
| **Train** | 7,800 | 80% | ~19,378 |
| **Test** | 1,950 | 20% | ~4,845 |
| **Total** | 9,750 | 100% | ~24,223 |

**Stratification:** None (random split)

**Validation Set:** None (could create 10% val from train for hyperparameter tuning)

### Additional Test Sets

**Day-Time Test (`daytest1/`):**
- **Purpose:** Evaluate RGB performance in good lighting
- **Images:** ~500-1000 (estimated)
- **Conditions:** Daytime, well-lit scenes

**Night-Time Test (`nighttest1/`):**
- **Purpose:** Evaluate Thermal/Event performance in low/no light
- **Images:** ~500-1000 (estimated)
- **Conditions:** Night-time, low-light scenes

**Usage:** Evaluate modality ablations (RGB-only vs. Thermal+Event vs. All)

---

## Data Augmentation

### Current Status

**Baseline:** Minimal augmentation (normalization only)

```python
# Normalization (applied in __getitem__)
image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
image_std = [0.229, 0.224, 0.225, 0.5, 0.5]

normalized_image = (image / 255.0 - mean) / std
```

**No Augmentation Applied:**
- Random horizontal flip
- Random resized crop
- Color jittering
- Cutout/Mixup
- Multi-scale training

### Proposed Augmentations (Future Work)

**For RGB channels:**
```python
transforms = [
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
]
```

**For Thermal/Event channels:**
- RandomHorizontalFlip (synchronized with RGB)
- Brightness jittering (limited, preserve physical meaning)
- No color jittering (single-channel)

**Multi-Modal Specific:**
- Synchronized augmentation across modalities (same flip/crop for all 5 channels)
- Modality dropout (randomly zero out channels during training for robustness)

---

## Dataset Statistics

### Image Statistics

**Total Files:** 10,489 .npy files

**Valid Annotations:** 9,750 (93%)

**Missing/Empty Labels:** 739 (7%)

**Resolution Distribution (estimated):**
| Resolution | Count (approx) | Percentage |
|------------|----------------|------------|
| 300×400 | ~3,500 | 33% |
| 320×240 | ~2,000 | 19% |
| 640×480 | ~1,500 | 14% |
| Other | ~3,500 | 34% |

### Bounding Box Statistics

**Total Boxes:** ~24,223

**Boxes per Image:**
- **Mean:** ~3.14
- **Median:** 3
- **Min:** 1
- **Max:** ~15 (estimated)

**Box Size Distribution (relative to image size):**
| Size Category | Area Range | Count (approx) | Percentage |
|---------------|------------|----------------|------------|
| Small | < 32² pixels | ~8,000 | 33% |
| Medium | 32² - 96² pixels | ~12,000 | 50% |
| Large | > 96² pixels | ~4,223 | 17% |

**Class Distribution:**
- **Class 0:** 24,223 boxes (100%)
- **Single-class dataset** (likely "object" or "pedestrian")

### Dataset Challenges

**1. Class Imbalance:**
- Single-class: No class imbalance, but limits detection diversity

**2. Scale Variation:**
- Objects range from small (distant) to large (nearby)
- FPN addresses this with multi-scale features

**3. Occlusion:**
- Likely present (urban/crowd scenes), not explicitly annotated

**4. Lighting Variation:**
- Day/night splits suggest significant lighting variation
- Thermal/Event modalities address this

**5. Motion:**
- Event camera suggests dynamic scenes
- Static vs. dynamic object detection

---

## Dataset Insights for CVPR Paper

### Key Points to Highlight

1. **Multi-Modal Richness:**
   - 5-channel input (RGB + Thermal + Event)
   - Complementary modalities for diverse conditions

2. **Real-World Complexity:**
   - Variable resolutions, lighting, occlusion
   - Day/night test sets for modality evaluation

3. **Scale Diversity:**
   - Small, medium, large objects
   - Tests FPN and multi-scale fusion

4. **Annotation Quality:**
   - 93% coverage (high quality)
   - YOLO format (easy to use)

### Limitations to Acknowledge

1. **Single Class:**
   - Limited to one object category
   - Generalization to multi-class requires further work

2. **No Validation Set:**
   - Test set used for final evaluation
   - Hyperparameter tuning risks overfitting to test

3. **Fixed Split:**
   - Single 80/20 split (no cross-validation)
   - Results sensitive to split

---

[← Back to Index](00_INDEX.md) | [← Previous: Architecture](02_ARCHITECTURE_DEEP_DIVE.md) | [Next: Ablation Studies →](04_ABLATION_STUDIES.md)
