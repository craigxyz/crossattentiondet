#!/usr/bin/env python
"""
Analyze and visualize CSSA ablation experiment results.

This script:
- Parses all experiment results
- Creates comparison tables
- Generates plots
- Exports publication-ready summaries

Usage:
    python crossattentiondet/ablations/scripts/analyze_cssa_results.py \
        --results-dir results/cssa_ablations \
        --output-dir results/cssa_ablations/analysis
"""

import os
import argparse
import csv
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class CSSAResultsAnalyzer:
    """Analyze and visualize CSSA ablation results."""

    def __init__(self, results_dir, output_dir):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.summary_csv = os.path.join(results_dir, 'summary_all_experiments.csv')

        os.makedirs(output_dir, exist_ok=True)

        # Load results
        self.results = self.load_results()

    def load_results(self):
        """Load results from summary CSV."""
        if not os.path.exists(self.summary_csv):
            raise FileNotFoundError(f"Summary CSV not found: {self.summary_csv}")

        results = []
        with open(self.summary_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                row['mAP'] = float(row['mAP'])
                row['mAP_50'] = float(row['mAP_50'])
                row['mAP_75'] = float(row['mAP_75'])
                row['mAP_small'] = float(row['mAP_small'])
                row['mAP_medium'] = float(row['mAP_medium'])
                row['mAP_large'] = float(row['mAP_large'])
                row['params_M'] = float(row['params_M'])
                row['time_hr'] = float(row['time_hr'])
                row['threshold'] = float(row['threshold'])
                row['kernel'] = int(row['kernel'])
                row['epochs'] = int(row['epochs'])
                results.append(row)

        print(f"Loaded {len(results)} experiment results")
        return results

    def create_stage_comparison_table(self):
        """Create table comparing different stage configurations (threshold=0.5)."""
        print("\nStage Configuration Comparison (threshold=0.5):")
        print("=" * 100)
        print(f"{'Stage Config':<15} {'Stages':<15} {'mAP':<8} {'AP50':<8} {'AP75':<8} {'Params(M)':<10} {'Time(hr)':<10}")
        print("=" * 100)

        # Filter for threshold=0.5
        stage_results = [r for r in self.results if r['threshold'] == 0.5]

        # Sort by mAP descending
        stage_results.sort(key=lambda x: x['mAP'], reverse=True)

        # Save to CSV
        output_csv = os.path.join(self.output_dir, 'stage_comparison.csv')
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Stage Config', 'Stages', 'mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'Params_M', 'Time_hr'])

            for rank, r in enumerate(stage_results, 1):
                print(f"{r['stage_label']:<15} {r['stages']:<15} {r['mAP']:<8.4f} {r['mAP_50']:<8.4f} {r['mAP_75']:<8.4f} {r['params_M']:<10.2f} {r['time_hr']:<10.2f}")
                writer.writerow([
                    rank, r['stage_label'], r['stages'], r['mAP'], r['mAP_50'], r['mAP_75'],
                    r['mAP_small'], r['mAP_medium'], r['mAP_large'], r['params_M'], r['time_hr']
                ])

        print("=" * 100)
        print(f"Saved to: {output_csv}")

        return stage_results

    def create_threshold_sensitivity_table(self):
        """Create table showing threshold sensitivity for top configs."""
        print("\nThreshold Sensitivity Analysis:")
        print("=" * 80)

        # Group by stage configuration
        by_stage = defaultdict(list)
        for r in self.results:
            by_stage[r['stage_label']].append(r)

        # Find configs with multiple thresholds
        multi_threshold_configs = {label: exps for label, exps in by_stage.items() if len(exps) > 1}

        if not multi_threshold_configs:
            print("No threshold variants found")
            return

        output_csv = os.path.join(self.output_dir, 'threshold_sensitivity.csv')
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stage Config', 'Threshold', 'mAP', 'AP50', 'AP75', 'Delta_mAP'])

            for label, exps in multi_threshold_configs.items():
                print(f"\n{label} (stages {exps[0]['stages']}):")
                print(f"  {'Threshold':<12} {'mAP':<8} {'AP50':<8} {'AP75':<8} {'Delta mAP':<10}")

                # Sort by threshold
                exps.sort(key=lambda x: x['threshold'])

                # Find baseline (threshold=0.5)
                baseline_mAP = next((e['mAP'] for e in exps if e['threshold'] == 0.5), exps[0]['mAP'])

                for e in exps:
                    delta_mAP = e['mAP'] - baseline_mAP
                    print(f"  {e['threshold']:<12.1f} {e['mAP']:<8.4f} {e['mAP_50']:<8.4f} {e['mAP_75']:<8.4f} {delta_mAP:+<10.4f}")
                    writer.writerow([label, e['threshold'], e['mAP'], e['mAP_50'], e['mAP_75'], delta_mAP])

        print("=" * 80)
        print(f"Saved to: {output_csv}")

    def plot_stage_comparison(self):
        """Plot mAP comparison across stage configurations."""
        # Filter for threshold=0.5
        stage_results = [r for r in self.results if r['threshold'] == 0.5]
        stage_results.sort(key=lambda x: x['mAP'], reverse=True)

        labels = [r['stage_label'] for r in stage_results]
        mAP = [r['mAP'] for r in stage_results]
        mAP_50 = [r['mAP_50'] for r in stage_results]
        mAP_75 = [r['mAP_75'] for r in stage_results]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, mAP, width, label='mAP@[.5:.95]', color='steelblue')
        ax.bar(x, mAP_50, width, label='mAP@.50', color='lightcoral')
        ax.bar(x + width, mAP_75, width, label='mAP@.75', color='lightgreen')

        ax.set_xlabel('Stage Configuration', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('CSSA Stage Ablation: mAP Comparison (threshold=0.5)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'mAP_by_stage_config.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: {output_path}")
        plt.close()

    def plot_threshold_sensitivity(self):
        """Plot threshold sensitivity for configs with multiple thresholds."""
        # Group by stage configuration
        by_stage = defaultdict(list)
        for r in self.results:
            by_stage[r['stage_label']].append(r)

        # Find configs with multiple thresholds
        multi_threshold_configs = {label: exps for label, exps in by_stage.items() if len(exps) > 1}

        if not multi_threshold_configs:
            print("No threshold variants to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for label, exps in multi_threshold_configs.items():
            # Sort by threshold
            exps.sort(key=lambda x: x['threshold'])

            thresholds = [e['threshold'] for e in exps]
            mAPs = [e['mAP'] for e in exps]

            ax.plot(thresholds, mAPs, marker='o', linewidth=2, markersize=8, label=f"{label} (stages {exps[0]['stages']})")

        ax.set_xlabel('Channel Switching Threshold', fontsize=12)
        ax.set_ylabel('mAP@[.5:.95]', fontsize=12)
        ax.set_title('CSSA Threshold Sensitivity Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'threshold_sensitivity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()

    def plot_size_specific_performance(self):
        """Plot size-specific AP (small, medium, large) for stage configs."""
        stage_results = [r for r in self.results if r['threshold'] == 0.5]
        stage_results.sort(key=lambda x: x['mAP'], reverse=True)

        labels = [r['stage_label'] for r in stage_results]
        small = [r['mAP_small'] for r in stage_results]
        medium = [r['mAP_medium'] for r in stage_results]
        large = [r['mAP_large'] for r in stage_results]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, small, width, label='AP Small', color='coral')
        ax.bar(x, medium, width, label='AP Medium', color='gold')
        ax.bar(x + width, large, width, label='AP Large', color='seagreen')

        ax.set_xlabel('Stage Configuration', fontsize=12)
        ax.set_ylabel('AP', fontsize=12)
        ax.set_title('CSSA Stage Ablation: Size-Specific Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'size_specific_performance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()

    def create_summary_report(self):
        """Create markdown summary report."""
        output_path = os.path.join(self.output_dir, 'RESULTS_SUMMARY.md')

        with open(output_path, 'w') as f:
            f.write("# CSSA Fusion Ablation Study - Results Summary\n\n")
            f.write(f"**Generated:** {os.popen('date').read().strip()}\n\n")
            f.write(f"**Total Experiments:** {len(self.results)}\n\n")

            # Best overall
            best = max(self.results, key=lambda x: x['mAP'])
            f.write("## Best Configuration\n\n")
            f.write(f"- **Stage Config:** {best['stage_label']} (stages {best['stages']})\n")
            f.write(f"- **Threshold:** {best['threshold']}\n")
            f.write(f"- **mAP:** {best['mAP']:.4f}\n")
            f.write(f"- **AP50:** {best['mAP_50']:.4f}\n")
            f.write(f"- **AP75:** {best['mAP_75']:.4f}\n\n")

            # Stage comparison (threshold=0.5)
            f.write("## Stage Configuration Comparison (threshold=0.5)\n\n")
            f.write("| Rank | Config | Stages | mAP | AP50 | AP75 | Params(M) | Time(hr) |\n")
            f.write("|------|--------|--------|-----|------|------|-----------|----------|\n")

            stage_results = [r for r in self.results if r['threshold'] == 0.5]
            stage_results.sort(key=lambda x: x['mAP'], reverse=True)

            for rank, r in enumerate(stage_results, 1):
                f.write(f"| {rank} | {r['stage_label']} | {r['stages']} | {r['mAP']:.4f} | {r['mAP_50']:.4f} | {r['mAP_75']:.4f} | {r['params_M']:.2f} | {r['time_hr']:.2f} |\n")

            # Threshold sensitivity
            by_stage = defaultdict(list)
            for r in self.results:
                by_stage[r['stage_label']].append(r)

            multi_threshold = {label: exps for label, exps in by_stage.items() if len(exps) > 1}

            if multi_threshold:
                f.write("\n## Threshold Sensitivity\n\n")
                for label, exps in multi_threshold.items():
                    f.write(f"### {label}\n\n")
                    f.write("| Threshold | mAP | AP50 | AP75 |\n")
                    f.write("|-----------|-----|------|------|\n")

                    exps.sort(key=lambda x: x['threshold'])
                    for e in exps:
                        f.write(f"| {e['threshold']:.1f} | {e['mAP']:.4f} | {e['mAP_50']:.4f} | {e['mAP_75']:.4f} |\n")
                    f.write("\n")

            # Key findings
            f.write("## Key Findings\n\n")
            best_stage = max(stage_results, key=lambda x: x['mAP'])
            worst_stage = min(stage_results, key=lambda x: x['mAP'])
            f.write(f"1. **Best stage config:** {best_stage['stage_label']} achieved mAP of {best_stage['mAP']:.4f}\n")
            f.write(f"2. **Worst stage config:** {worst_stage['stage_label']} achieved mAP of {worst_stage['mAP']:.4f}\n")
            f.write(f"3. **Performance gap:** {(best_stage['mAP'] - worst_stage['mAP'])*100:.2f}% absolute difference\n")

            # Training time
            avg_time = np.mean([r['time_hr'] for r in self.results])
            f.write(f"4. **Average training time:** {avg_time:.2f} hours per experiment\n")
            f.write(f"5. **Total compute time:** {sum(r['time_hr'] for r in self.results):.2f} hours\n\n")

            f.write("## Visualizations\n\n")
            f.write("- `mAP_by_stage_config.png` - Stage-wise mAP comparison\n")
            f.write("- `threshold_sensitivity.png` - Threshold sensitivity curves\n")
            f.write("- `size_specific_performance.png` - Performance by object size\n\n")

        print(f"\nSaved summary report: {output_path}")

    def analyze_all(self):
        """Run all analyses."""
        print("\n" + "=" * 80)
        print("CSSA ABLATION RESULTS ANALYSIS")
        print("=" * 80)

        self.create_stage_comparison_table()
        self.create_threshold_sensitivity_table()

        print("\nGenerating plots...")
        self.plot_stage_comparison()
        self.plot_threshold_sensitivity()
        self.plot_size_specific_performance()

        self.create_summary_report()

        print("\n" + "=" * 80)
        print(f"Analysis complete! Results saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze CSSA Ablation Results')

    parser.add_argument('--results-dir', type=str, default='results/cssa_ablations',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for analysis (default: results-dir/analysis)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')

    analyzer = CSSAResultsAnalyzer(args.results_dir, args.output_dir)
    analyzer.analyze_all()


if __name__ == '__main__':
    main()
