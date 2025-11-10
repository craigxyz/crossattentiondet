#!/usr/bin/env python3
"""Add baseline experiment results to the comprehensive report."""

# Parse the baseline evaluation data from the checkpoint directory
baseline_results = {
    'mit_b0': {
        'backbone': 'mit_b0',
        'params_M': 27.8,
        'size_mb': 106.00,
        'inference_ms': 42.59,
        'fps': 23.48,
        'mAP_50': 0.9391,
        'mAP': 0.6465,
        'epochs': 1,
        'training_time_min': 22.91
    },
    'mit_b1': {
        'backbone': 'mit_b1',
        'params_M': 60.0,
        'size_mb': 228.95,
        'inference_ms': 54.54,
        'fps': 18.34,
        'mAP_50': 0.9490,
        'mAP': 0.6624,
        'epochs': 1,
        'training_time_min': 26.42
    },
    'mit_b2': {
        'backbone': 'mit_b2',
        'params_M': 82.1,
        'size_mb': 313.22,
        'inference_ms': 78.57,
        'fps': 12.73,
        'mAP_50': 0.9440,
        'mAP': 0.6242,
        'epochs': 1,
        'training_time_min': 37.96
    },
    'mit_b4': {
        'backbone': 'mit_b4',
        'params_M': 155.4,
        'size_mb': 592.81,
        'inference_ms': 145.83,
        'fps': 6.86,
        'mAP_50': 0.9315,
        'mAP': 0.6270,
        'epochs': 1,
        'training_time_min': 64.87
    },
    'mit_b5': {
        'backbone': 'mit_b5',
        'params_M': 196.6,
        'size_mb': 749.98,
        'inference_ms': 175.21,
        'fps': 5.71,
        'mAP_50': 0.9402,
        'mAP': 0.6318,
        'epochs': 1,
        'training_time_min': 74.35
    }
}

# Create baseline section markdown
def generate_baseline_section():
    md = []
    md.append("## 6. Baseline Models (CMX with FRM+FFM)")
    md.append("")
    md.append("**Fusion Method:** Original CMX with Feature Rectify Module (FRM) and Feature Fusion Module (FFM)")
    md.append("")
    md.append("**Training:** 1 epoch, batch_size=2, lr=0.005")
    md.append("")
    md.append("**Important Note:** These baseline models were trained for only 1 epoch for initial validation. ")
    md.append("For fair comparison with ablation studies (15 epochs), these would need to be retrained.")
    md.append("")
    md.append("| Backbone | Params (M) | mAP | mAP@50 | Size (MB) | Inference (ms) | FPS | Training (min) |")
    md.append("|----------|------------|-----|--------|-----------|----------------|-----|----------------|")

    # Sort by mAP descending
    sorted_baselines = sorted(baseline_results.items(), key=lambda x: x[1]['mAP'], reverse=True)

    for name, data in sorted_baselines:
        md.append(f"| {data['backbone']} | {data['params_M']:.1f} | "
                 f"{data['mAP']:.4f} | {data['mAP_50']:.4f} | "
                 f"{data['size_mb']:.2f} | {data['inference_ms']:.2f} | "
                 f"{data['fps']:.2f} | {data['training_time_min']:.2f} |")

    md.append("")
    best = sorted_baselines[0][1]
    md.append(f"**Best Baseline (1 epoch):** {best['backbone']} with mAP={best['mAP']:.4f} ({best['mAP']*100:.2f}%)")
    md.append("")
    md.append("### Performance vs Ablation Studies")
    md.append("")
    md.append("**⚠️ Training Epoch Mismatch:**")
    md.append("- Baseline models: **1 epoch** → mAP ~0.66 (66%)")
    md.append("- GAFF ablations: **15 epochs** → mAP ~0.84 (84%)")
    md.append("- CSSA ablations: **15 epochs** → mAP ~0.84 (84%)")
    md.append("- Modality ablations: **15 epochs** → mAP ~0.83 (83%)")
    md.append("")
    md.append("**Fair comparison requires:** Retraining baseline models with 15 epochs to match ablation study protocols.")
    md.append("")
    md.append("### Speed-Accuracy Trade-off")
    md.append("")
    md.append("**Fastest:** mit_b0 (23.48 FPS, mAP=0.6465)")
    md.append("")
    md.append("**Best Accuracy:** mit_b1 (18.34 FPS, mAP=0.6624)")
    md.append("")
    md.append("**Slowest:** mit_b5 (5.71 FPS, mAP=0.6318)")
    md.append("")
    md.append("---")
    md.append("")

    return "\n".join(md)

if __name__ == "__main__":
    baseline_section = generate_baseline_section()

    # Read the current comprehensive results
    with open("COMPREHENSIVE_EXPERIMENTAL_RESULTS.md", "r") as f:
        content = f.read()

    # Find where to insert (before "## 7. Key Findings" or after GAFF Pilot)
    insert_marker = "## 7. Key Findings"

    if insert_marker in content:
        parts = content.split(insert_marker)
        updated_content = parts[0] + baseline_section + insert_marker + parts[1]

        # Update section numbers that come after
        updated_content = updated_content.replace("## 7. Key Findings", "## 7. Key Findings")
        updated_content = updated_content.replace("## 8. Experimental Setup", "## 8. Experimental Setup")

        # Actually we need to renumber
        # 6 becomes the new baseline section
        # 7 stays as key findings
        # 8 stays as experimental setup

        with open("COMPREHENSIVE_EXPERIMENTAL_RESULTS.md", "w") as f:
            f.write(updated_content)

        print("✓ Added baseline section to COMPREHENSIVE_EXPERIMENTAL_RESULTS.md")
        print(f"  Baseline models: 5")
        print(f"  Best baseline: mit_b1 (mAP=0.6624, 1 epoch)")
    else:
        print("Error: Could not find insertion point")
