#!/usr/bin/env python
"""
Training script for CrossAttentionDet.

Usage:
    python scripts/train.py --data data/ --epochs 15 --batch-size 2
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crossattentiondet.config import Config
from crossattentiondet.training.trainer import Trainer


def main():
    # Load configuration from command line
    config = Config.from_args()

    print("=== CrossAttentionDet Training ===")
    print(config)

    # Create trainer and run training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
