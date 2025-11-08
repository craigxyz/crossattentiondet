"""
Pre-GPU Deployment Checklist

Automated verification script that runs all tests before GPU deployment.
This ensures error-free code before expensive GPU training begins.

Run with: python -m crossattentiondet.ablations.scripts.pre_gpu_checklist
"""

import sys
import os
import subprocess
import time

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        elapsed = time.time() - start_time

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        success = result.returncode == 0

        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"\n{status} ({elapsed:.1f}s)\n")

        return success, elapsed

    except subprocess.TimeoutExpired:
        print(f"\n✗ TIMEOUT (exceeded 300s)\n")
        return False, 300.0
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        return False, 0.0


def main():
    """Run pre-GPU checklist."""
    print("="*80)
    print("GAFF Pre-GPU Deployment Checklist")
    print("="*80)
    print("\nThis script will verify that all GAFF components are working correctly")
    print("before deploying to expensive GPU resources.\n")

    results = []

    # Test 1: GAFF module unit tests
    success, elapsed = run_command(
        "python -m crossattentiondet.ablations.fusion.test_gaff",
        "GAFF Module Unit Tests"
    )
    results.append(("GAFF Module Unit Tests", success, elapsed))

    # Test 2: GAFF module verification
    success, elapsed = run_command(
        "python -m crossattentiondet.ablations.fusion.verify_gaff",
        "GAFF Module Verification"
    )
    results.append(("GAFF Module Verification", success, elapsed))

    # Test 3: Encoder verification
    success, elapsed = run_command(
        "python -m crossattentiondet.ablations.scripts.verify_gaff_encoder",
        "GAFF Encoder Verification"
    )
    results.append(("GAFF Encoder Verification", success, elapsed))

    # Test 4: Dry-run tests
    success, elapsed = run_command(
        "python -m crossattentiondet.ablations.scripts.dry_run_gaff",
        "Dry-Run Training Tests"
    )
    results.append(("Dry-Run Training Tests", success, elapsed))

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Test':<40s} {'Status':<12s} {'Time (s)':<10s}")
    print("-"*80)

    all_passed = True
    total_time = 0

    for test_name, success, elapsed in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<40s} {status:<12s} {elapsed:>8.1f}")
        all_passed = all_passed and success
        total_time += elapsed

    print("-"*80)
    print(f"{'Total':<40s} {'':<12s} {total_time:>8.1f}")
    print()

    passed_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)

    print(f"Tests passed: {passed_count}/{total_count}")
    print()

    if all_passed:
        print("="*80)
        print("✓ ✓ ✓  ALL CHECKS PASSED  ✓ ✓ ✓")
        print("="*80)
        print()
        print("Your GAFF implementation is ready for GPU deployment!")
        print()
        print("Next steps:")
        print("  1. Transfer code to GPU system")
        print("  2. Set up dataset paths")
        print("  3. Run training with: python -m crossattentiondet.ablations.scripts.train_gaff_ablation")
        print("  4. Or run full ablation suite with: python -m crossattentiondet.ablations.scripts.run_gaff_ablations")
        print()
        print("="*80)
        return 0
    else:
        print("="*80)
        print("✗ ✗ ✗  SOME CHECKS FAILED  ✗ ✗ ✗")
        print("="*80)
        print()
        print("Please fix the failing tests before deploying to GPU.")
        print("Review the error messages above for details.")
        print()
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
