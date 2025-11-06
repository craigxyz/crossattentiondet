#!/usr/bin/env python
"""
Testing/evaluation script for CrossAttentionDet.

Usage:
    python scripts/test.py --data data/ --model crossattentiondet.pth
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crossattentiondet.config import Config
from crossattentiondet.training.evaluator import Evaluator


def main():
    # Load configuration from command line
    config = Config.from_args()

    print("=== CrossAttentionDet Evaluation ===")
    print(config)

    # Create evaluator and run evaluation
    evaluator = Evaluator(config)
    evaluator.load_checkpoint()
    results = evaluator.evaluate()

    if results:
        print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
