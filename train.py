#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training entry point
Simplified entry based on modular design, calls Trainer for training
"""

from config.args import build_argparser
from training.trainer import Trainer


if __name__ == '__main__':
    # Parse command line arguments
    parser = build_argparser()
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()
