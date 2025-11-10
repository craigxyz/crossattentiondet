#!/usr/bin/env python3
"""Compile all experimental results into a comprehensive markdown report."""

import json
import csv
from pathlib import Path
from datetime import datetime

def load_json_result(json_path):
    """Load results from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def format_number(val, decimals=4):
    """Format a number with specified decimals."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except:
        return str(val)

def compile_results():
    """Compile all results from various experiments."""

    results = {
        'gaff_ablations': [],
        'cssa_ablations': [],
        'cssa_ablations_phase2': [],
        'modality_ablations': [],
        'gaff_pilot': None,
        'baseline': None
    }

    # 1. GAFF Ablations - from CSV
    gaff_csv = Path('results/gaff_ablations_full/summary_all_experiments.csv')
    if gaff_csv.exists():
        with open(gaff_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['experiment_id']:  # Skip empty rows
                    results['gaff_ablations'].append(row)

    # 2. GAFF Ablations - also get JSON for more details
    gaff_dirs = [
        'results/gaff_ablations_full/phase1_stage_selection',
        'results/gaff_ablations_full/phase2_hyperparameter_tuning'
    ]

    gaff_json_data = {}
    for gaff_dir in gaff_dirs:
        gaff_path = Path(gaff_dir)
        if gaff_path.exists():
            for exp_dir in sorted(gaff_path.iterdir()):
                if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                    json_file = exp_dir / 'final_results.json'
                    if json_file.exists():
                        data = load_json_result(json_file)
                        if data:
                            gaff_json_data[exp_dir.name] = data

    # 3. CSSA Ablations Phase 1
    cssa_path = Path('results/cssa_ablations')
    if cssa_path.exists():
        for exp_dir in sorted(cssa_path.iterdir()):
            if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                json_file = exp_dir / 'final_results.json'
                if json_file.exists():
                    data = load_json_result(json_file)
                    if data:
                        results['cssa_ablations'].append(data)

    # 4. CSSA Ablations Phase 2
    cssa2_path = Path('results/cssa_ablations_phase2')
    if cssa2_path.exists():
        for exp_dir in sorted(cssa2_path.iterdir()):
            if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                json_file = exp_dir / 'final_results.json'
                if json_file.exists():
                    data = load_json_result(json_file)
                    if data:
                        results['cssa_ablations_phase2'].append(data)

    # 5. Modality Ablations
    mod_path = Path('results/modality_ablations')
    if mod_path.exists():
        for exp_dir in sorted(mod_path.iterdir()):
            if exp_dir.is_dir():
                json_file = exp_dir / 'final_results.json'
                if json_file.exists():
                    data = load_json_result(json_file)
                    if data:
                        data['modality_name'] = exp_dir.name
                        results['modality_ablations'].append(data)

    # 6. GAFF Pilot
    pilot_path = Path('gaff_pilot_5epochs/final_results.json')
    if pilot_path.exists():
        results['gaff_pilot'] = load_json_result(pilot_path)

    # 7. Baseline
    baseline_path = Path('results/baseline_mit_b0_evaluated')
    if baseline_path.exists():
        json_file = baseline_path / 'final_results.json'
        if json_file.exists():
            results['baseline'] = load_json_result(json_file)

    return results, gaff_json_data

def generate_markdown_report(results, gaff_json_data):
    """Generate comprehensive markdown report."""

    md = []
    md.append("# Comprehensive Experimental Results Report")
    md.append("")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("---")
    md.append("")

    # Summary Statistics
    md.append("## Executive Summary")
    md.append("")
    total_experiments = (
        len(results['gaff_ablations']) +
        len(results['cssa_ablations']) +
        len(results['cssa_ablations_phase2']) +
        len(results['modality_ablations']) +
        (1 if results['gaff_pilot'] else 0) +
        (1 if results['baseline'] else 0)
    )
    md.append(f"**Total Experiments Completed:** {total_experiments}")
    md.append(f"- GAFF Ablations: {len(results['gaff_ablations'])}")
    md.append(f"- CSSA Ablations (Phase 1): {len(results['cssa_ablations'])}")
    md.append(f"- CSSA Ablations (Phase 2): {len(results['cssa_ablations_phase2'])}")
    md.append(f"- Modality Ablations: {len(results['modality_ablations'])}")
    md.append(f"- GAFF Pilot: {1 if results['gaff_pilot'] else 0}")
    md.append(f"- Baseline: {1 if results['baseline'] else 0}")
    md.append("")
    md.append("---")
    md.append("")

    # GAFF Ablations
    if results['gaff_ablations']:
        md.append("## 1. GAFF Ablation Study")
        md.append("")
        md.append("**Fusion Method:** GAFF (Gated Attention Feature Fusion)")
        md.append("")
        md.append("**Objective:** Evaluate different stage configurations and hyperparameters for GAFF fusion")
        md.append("")

        # Phase 1
        phase1 = [r for r in results['gaff_ablations'] if r['phase'] == '1']
        if phase1:
            md.append("### Phase 1: Stage Selection (8 experiments)")
            md.append("")
            md.append("Default hyperparameters: SE_reduction=4, inter_shared=False, merge_bottleneck=False")
            md.append("")
            md.append("| Exp ID | Stages | mAP | mAP@50 | mAP@75 | Train Loss | Time (h) | Params (M) |")
            md.append("|--------|--------|-----|--------|--------|------------|----------|------------|")
            for r in phase1:
                md.append(f"| {r['experiment_id']} | {r['stages']} | {format_number(r['mAP'])} | "
                         f"{format_number(r['mAP_50'])} | {format_number(r['mAP_75'])} | "
                         f"{format_number(r['final_train_loss'])} | {format_number(r['time_hours'], 2)} | "
                         f"{format_number(r['total_params_M'], 2)} |")
            md.append("")

            # Find best from phase 1
            best_p1 = max(phase1, key=lambda x: float(x['mAP']))
            md.append(f"**Best Phase 1 Result:** {best_p1['experiment_id']} (Stages: {best_p1['stages']}) "
                     f"with mAP={format_number(best_p1['mAP'])}")
            md.append("")

        # Phase 2
        phase2 = [r for r in results['gaff_ablations'] if r['phase'] == '2']
        if phase2:
            md.append("### Phase 2: Hyperparameter Tuning")
            md.append("")
            md.append("Tuning SE_reduction, inter_shared, and merge_bottleneck parameters")
            md.append("")
            md.append("| Exp ID | Stages | SE_r | Inter_S | Merge_B | mAP | mAP@50 | mAP@75 | Loss | Time (h) | Params (M) |")
            md.append("|--------|--------|------|---------|---------|-----|--------|--------|------|----------|------------|")
            for r in phase2:
                md.append(f"| {r['experiment_id']} | {r['stages']} | {r['se_reduction']} | "
                         f"{r['inter_shared']} | {r['merge_bottleneck']} | {format_number(r['mAP'])} | "
                         f"{format_number(r['mAP_50'])} | {format_number(r['mAP_75'])} | "
                         f"{format_number(r['final_train_loss'])} | {format_number(r['time_hours'], 2)} | "
                         f"{format_number(r['total_params_M'], 2)} |")
            md.append("")

            # Find best overall
            best_overall = max(results['gaff_ablations'], key=lambda x: float(x['mAP']))
            md.append(f"**Best Overall GAFF Result:** {best_overall['experiment_id']} "
                     f"(Stages: {best_overall['stages']}, SE_r={best_overall['se_reduction']}, "
                     f"Inter_S={best_overall['inter_shared']}, Merge_B={best_overall['merge_bottleneck']}) "
                     f"with mAP={format_number(best_overall['mAP'])}")
            md.append("")

        # Detailed results with more metrics
        if gaff_json_data:
            md.append("### GAFF Detailed Metrics")
            md.append("")
            md.append("Additional metrics from completed experiments:")
            md.append("")
            md.append("| Exp ID | mAP_small | mAP_medium | mAP_large | Epochs | Batch Size | LR |")
            md.append("|--------|-----------|------------|-----------|--------|------------|-----|")

            for exp_id, data in sorted(gaff_json_data.items()):
                if 'results' in data and 'config' in data:
                    res = data['results']
                    cfg = data['config']
                    md.append(f"| {exp_id} | "
                             f"{format_number(res.get('mAP_small', 'N/A'))} | "
                             f"{format_number(res.get('mAP_medium', 'N/A'))} | "
                             f"{format_number(res.get('mAP_large', 'N/A'))} | "
                             f"{cfg.get('epochs', 'N/A')} | "
                             f"{cfg.get('batch_size', 'N/A')} | "
                             f"{cfg.get('learning_rate', 'N/A')} |")
            md.append("")

        md.append("---")
        md.append("")

    # CSSA Ablations Phase 1
    if results['cssa_ablations']:
        md.append("## 2. CSSA Ablation Study - Phase 1")
        md.append("")
        md.append("**Fusion Method:** CSSA (Channel Switching and Spatial Attention)")
        md.append("")
        md.append("**Objective:** Evaluate CSSA fusion at different backbone stages with threshold=0.5")
        md.append("")
        md.append("| Exp ID | Stages | Threshold | mAP | mAP@50 | mAP@75 | mAP_small | mAP_medium | mAP_large | Loss | Time (h) | Params (M) |")
        md.append("|--------|--------|-----------|-----|--------|--------|-----------|------------|-----------|------|----------|------------|")

        for data in results['cssa_ablations']:
            if 'results' in data and 'config' in data:
                res = data['results']
                cfg = data['config']
                exp_id = data.get('experiment_id', 'N/A')
                md.append(f"| {exp_id} | "
                         f"{cfg.get('cssa_stages', 'N/A')} | "
                         f"{cfg.get('cssa_threshold', 'N/A')} | "
                         f"{format_number(res.get('mAP', 'N/A'))} | "
                         f"{format_number(res.get('mAP_50', 'N/A'))} | "
                         f"{format_number(res.get('mAP_75', 'N/A'))} | "
                         f"{format_number(res.get('mAP_small', 'N/A'))} | "
                         f"{format_number(res.get('mAP_medium', 'N/A'))} | "
                         f"{format_number(res.get('mAP_large', 'N/A'))} | "
                         f"{format_number(data.get('training', {}).get('final_train_loss', 'N/A'))} | "
                         f"{format_number(data.get('training', {}).get('total_time_hours', 'N/A'), 2)} | "
                         f"{format_number(data.get('model', {}).get('total_params_M', 'N/A'), 2)} |")
        md.append("")

        # Find best
        best_cssa = max(results['cssa_ablations'],
                       key=lambda x: float(x.get('results', {}).get('mAP', 0)))
        best_map = best_cssa.get('results', {}).get('mAP', 'N/A')
        best_id = best_cssa.get('experiment_id', 'N/A')
        best_stages = best_cssa.get('config', {}).get('cssa_stages', 'N/A')
        md.append(f"**Best CSSA Phase 1 Result:** {best_id} (Stages: {best_stages}) "
                 f"with mAP={format_number(best_map)}")
        md.append("")
        md.append("---")
        md.append("")

    # CSSA Ablations Phase 2
    if results['cssa_ablations_phase2']:
        md.append("## 3. CSSA Ablation Study - Phase 2")
        md.append("")
        md.append("**Objective:** Fine-tune CSSA threshold parameter on best stage configurations")
        md.append("")
        md.append("| Exp ID | Stages | Threshold | mAP | mAP@50 | mAP@75 | mAP_small | mAP_medium | mAP_large | Loss | Time (h) | Params (M) |")
        md.append("|--------|--------|-----------|-----|--------|--------|-----------|------------|-----------|------|----------|------------|")

        for data in results['cssa_ablations_phase2']:
            if 'results' in data and 'config' in data:
                res = data['results']
                cfg = data['config']
                exp_id = data.get('experiment_id', 'N/A')
                md.append(f"| {exp_id} | "
                         f"{cfg.get('cssa_stages', 'N/A')} | "
                         f"{cfg.get('cssa_threshold', 'N/A')} | "
                         f"{format_number(res.get('mAP', 'N/A'))} | "
                         f"{format_number(res.get('mAP_50', 'N/A'))} | "
                         f"{format_number(res.get('mAP_75', 'N/A'))} | "
                         f"{format_number(res.get('mAP_small', 'N/A'))} | "
                         f"{format_number(res.get('mAP_medium', 'N/A'))} | "
                         f"{format_number(res.get('mAP_large', 'N/A'))} | "
                         f"{format_number(data.get('training', {}).get('final_train_loss', 'N/A'))} | "
                         f"{format_number(data.get('training', {}).get('total_time_hours', 'N/A'), 2)} | "
                         f"{format_number(data.get('model', {}).get('total_params_M', 'N/A'), 2)} |")
        md.append("")

        # Find best
        best_cssa2 = max(results['cssa_ablations_phase2'],
                        key=lambda x: float(x.get('results', {}).get('mAP', 0)))
        best_map = best_cssa2.get('results', {}).get('mAP', 'N/A')
        best_id = best_cssa2.get('experiment_id', 'N/A')
        best_stages = best_cssa2.get('config', {}).get('cssa_stages', 'N/A')
        best_thresh = best_cssa2.get('config', {}).get('cssa_threshold', 'N/A')
        md.append(f"**Best CSSA Phase 2 Result:** {best_id} (Stages: {best_stages}, Threshold: {best_thresh}) "
                 f"with mAP={format_number(best_map)}")
        md.append("")
        md.append("---")
        md.append("")

    # Modality Ablations
    if results['modality_ablations']:
        md.append("## 4. Modality Ablation Study")
        md.append("")
        md.append("**Objective:** Evaluate model performance with different modality combinations")
        md.append("")
        md.append("| Modality Pair | mAP | mAP@50 | mAP@75 | mAP_small | mAP_medium | mAP_large | Loss | Time (h) | Params (M) | Backbone |")
        md.append("|---------------|-----|--------|--------|-----------|------------|-----------|------|----------|------------|----------|")

        for data in results['modality_ablations']:
            # Modality ablations use 'final_test_results' instead of 'results'
            res = data.get('final_test_results', data.get('results', {}))
            cfg = data.get('config', {})
            name = data.get('modality_name', data.get('experiment_id', 'N/A'))
            md.append(f"| {name.replace('_', ' + ').upper()} | "
                     f"{format_number(res.get('mAP', 'N/A'))} | "
                     f"{format_number(res.get('mAP_50', 'N/A'))} | "
                     f"{format_number(res.get('mAP_75', 'N/A'))} | "
                     f"{format_number(res.get('mAP_small', 'N/A'))} | "
                     f"{format_number(res.get('mAP_medium', 'N/A'))} | "
                     f"{format_number(res.get('mAP_large', 'N/A'))} | "
                     f"{format_number(data.get('training', {}).get('best_train_loss', 'N/A'))} | "
                     f"{format_number(data.get('training', {}).get('total_time_hours', 'N/A'), 2)} | "
                     f"{format_number(data.get('model', {}).get('total_params_M', 'N/A'), 2)} | "
                     f"{cfg.get('backbone', 'N/A')} |")
        md.append("")

        # Find best
        best_mod = max(results['modality_ablations'],
                      key=lambda x: float(x.get('final_test_results', x.get('results', {})).get('mAP', 0)))
        best_map = best_mod.get('final_test_results', best_mod.get('results', {})).get('mAP', 'N/A')
        best_name = best_mod.get('modality_name', best_mod.get('experiment_id', 'N/A'))
        md.append(f"**Best Modality Combination:** {best_name.replace('_', ' + ').upper()} "
                 f"with mAP={format_number(best_map)}")
        md.append("")
        md.append("---")
        md.append("")

    # GAFF Pilot
    if results['gaff_pilot']:
        md.append("## 5. GAFF Pilot Study")
        md.append("")
        md.append("**Objective:** 5-epoch pilot to validate GAFF implementation")
        md.append("")
        data = results['gaff_pilot']
        if 'results' in data and 'config' in data:
            res = data['results']
            cfg = data['config']

            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            md.append(f"| Experiment ID | {data.get('experiment_id', 'N/A')} |")
            md.append(f"| Epochs | {cfg.get('epochs', 'N/A')} |")
            md.append(f"| Backbone | {cfg.get('backbone', 'N/A')} |")
            md.append(f"| GAFF Stages | {cfg.get('gaff_stages', 'N/A')} |")
            md.append(f"| mAP | {format_number(res.get('mAP', 'N/A'))} |")
            md.append(f"| mAP@50 | {format_number(res.get('mAP_50', 'N/A'))} |")
            md.append(f"| mAP@75 | {format_number(res.get('mAP_75', 'N/A'))} |")
            md.append(f"| mAP Small | {format_number(res.get('mAP_small', 'N/A'))} |")
            md.append(f"| mAP Medium | {format_number(res.get('mAP_medium', 'N/A'))} |")
            md.append(f"| mAP Large | {format_number(res.get('mAP_large', 'N/A'))} |")
            md.append(f"| Final Train Loss | {format_number(data.get('training', {}).get('final_train_loss', 'N/A'))} |")
            md.append(f"| Training Time (hours) | {format_number(data.get('training', {}).get('total_time_hours', 'N/A'), 2)} |")
            md.append(f"| Total Parameters (M) | {format_number(data.get('model', {}).get('total_params_M', 'N/A'), 2)} |")
            md.append("")

        md.append("---")
        md.append("")

    # Baseline
    if results['baseline']:
        md.append("## 6. Baseline Model")
        md.append("")
        data = results['baseline']
        if 'results' in data:
            res = data['results']
            cfg = data.get('config', {})

            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            md.append(f"| Backbone | {cfg.get('backbone', 'N/A')} |")
            md.append(f"| mAP | {format_number(res.get('mAP', 'N/A'))} |")
            md.append(f"| mAP@50 | {format_number(res.get('mAP_50', 'N/A'))} |")
            md.append(f"| mAP@75 | {format_number(res.get('mAP_75', 'N/A'))} |")
            md.append("")

        md.append("---")
        md.append("")

    # Analysis Section
    md.append("## 7. Key Findings")
    md.append("")

    # Find overall best
    all_results_with_map = []

    for r in results['gaff_ablations']:
        all_results_with_map.append(('GAFF', r['experiment_id'], float(r['mAP'])))

    for data in results['cssa_ablations']:
        if 'results' in data:
            all_results_with_map.append(('CSSA-P1', data.get('experiment_id', 'N/A'),
                                        float(data['results'].get('mAP', 0))))

    for data in results['cssa_ablations_phase2']:
        if 'results' in data:
            all_results_with_map.append(('CSSA-P2', data.get('experiment_id', 'N/A'),
                                        float(data['results'].get('mAP', 0))))

    for data in results['modality_ablations']:
        res = data.get('final_test_results', data.get('results', {}))
        if res:
            name = data.get('modality_name', data.get('experiment_id', 'N/A'))
            all_results_with_map.append(('Modality', name,
                                        float(res.get('mAP', 0))))

    if all_results_with_map:
        all_results_with_map.sort(key=lambda x: x[2], reverse=True)

        md.append("### Top 10 Performing Experiments (by mAP)")
        md.append("")
        md.append("| Rank | Study | Experiment | mAP |")
        md.append("|------|-------|------------|-----|")

        for i, (study, exp_id, map_val) in enumerate(all_results_with_map[:10], 1):
            md.append(f"| {i} | {study} | {exp_id} | {format_number(map_val)} |")
        md.append("")

    md.append("### Insights")
    md.append("")
    md.append("#### GAFF Ablations:")
    if results['gaff_ablations']:
        md.append(f"- Total configurations tested: {len(results['gaff_ablations'])}")
        map_values = [float(r['mAP']) for r in results['gaff_ablations']]
        md.append(f"- mAP range: {format_number(min(map_values))} to {format_number(max(map_values))}")
        md.append(f"- Average mAP: {format_number(sum(map_values)/len(map_values))}")

    md.append("")
    md.append("#### CSSA Ablations:")
    if results['cssa_ablations'] or results['cssa_ablations_phase2']:
        total_cssa = len(results['cssa_ablations']) + len(results['cssa_ablations_phase2'])
        md.append(f"- Total configurations tested: {total_cssa}")
        cssa_maps = []
        for data in results['cssa_ablations'] + results['cssa_ablations_phase2']:
            if 'results' in data:
                cssa_maps.append(float(data['results'].get('mAP', 0)))
        if cssa_maps:
            md.append(f"- mAP range: {format_number(min(cssa_maps))} to {format_number(max(cssa_maps))}")
            md.append(f"- Average mAP: {format_number(sum(cssa_maps)/len(cssa_maps))}")

    md.append("")
    md.append("#### Modality Combinations:")
    if results['modality_ablations']:
        md.append(f"- Total combinations tested: {len(results['modality_ablations'])}")
        mod_maps = []
        for data in results['modality_ablations']:
            res = data.get('final_test_results', data.get('results', {}))
            if res and res.get('mAP'):
                mod_maps.append(float(res.get('mAP', 0)))
        if mod_maps:
            md.append(f"- mAP range: {format_number(min(mod_maps))} to {format_number(max(mod_maps))}")
            md.append(f"- Average mAP: {format_number(sum(mod_maps)/len(mod_maps))}")

    md.append("")
    md.append("---")
    md.append("")

    # Methodology
    md.append("## 8. Experimental Setup")
    md.append("")
    md.append("### Common Configuration")
    md.append("")
    md.append("- **Dataset:** RGBX Object Detection")
    md.append("- **Training Images:** 7,800")
    md.append("- **Test Images:** 1,950")
    md.append("- **Batch Size:** 16 (most experiments)")
    md.append("- **Learning Rate:** 0.02")
    md.append("- **Epochs:** 15 (main studies), 5 (pilot)")
    md.append("- **Optimizer:** SGD with momentum")
    md.append("- **Backbone:** MixTransformer (mit_b1, mit_b0)")
    md.append("")
    md.append("### Fusion Methods Evaluated")
    md.append("")
    md.append("1. **GAFF (Gated Attention Feature Fusion)**")
    md.append("   - Gated attention mechanism for multi-modal fusion")
    md.append("   - Tested at different encoder stages (1, 2, 3, 4, and combinations)")
    md.append("   - Hyperparameters: SE reduction ratio, inter-modality sharing, merge bottleneck")
    md.append("")
    md.append("2. **CSSA (Channel Switching and Spatial Attention)**")
    md.append("   - Channel-wise attention with threshold-based switching")
    md.append("   - Spatial attention using max and average pooling")
    md.append("   - Tested thresholds: 0.3, 0.5, 0.7")
    md.append("")
    md.append("---")
    md.append("")

    # Footer
    md.append("## Notes")
    md.append("")
    md.append("- All experiments used CUDA GPU acceleration")
    md.append("- Training time varies based on stage configuration and hyperparameters")
    md.append("- Results are reported on the test set after final epoch")
    md.append("- mAP metrics follow COCO evaluation protocol")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")

    return "\n".join(md)

if __name__ == "__main__":
    print("Compiling experimental results...")
    results, gaff_json_data = compile_results()

    print(f"Found:")
    print(f"  - GAFF ablations: {len(results['gaff_ablations'])}")
    print(f"  - CSSA ablations (P1): {len(results['cssa_ablations'])}")
    print(f"  - CSSA ablations (P2): {len(results['cssa_ablations_phase2'])}")
    print(f"  - Modality ablations: {len(results['modality_ablations'])}")
    print(f"  - GAFF pilot: {'Yes' if results['gaff_pilot'] else 'No'}")
    print(f"  - Baseline: {'Yes' if results['baseline'] else 'No'}")

    print("\nGenerating markdown report...")
    markdown = generate_markdown_report(results, gaff_json_data)

    output_file = "COMPREHENSIVE_EXPERIMENTAL_RESULTS.md"
    with open(output_file, 'w') as f:
        f.write(markdown)

    print(f"\n✓ Report saved to: {output_file}")
    print(f"  Total lines: {len(markdown.split(chr(10)))}")
