# Copyright (c) Phigent Robotics. All rights reserved.
"""Thin training launcher of the migrated BEVDet."""
import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train BEVDet (mmdet3d 1.4)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp', action='store_true', help='enable automatic-mixed-precision')
    parser.add_argument('--resume', action='store_true', help='resume from ckpt')
    parser.add_argument('--max-iters', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
    cfg.resume = args.resume
    if args.max_iters is not None:
        cfg.train_cfg = dict(
            by_epoch=False,
            max_iters=args.max_iters,
            val_interval=args.max_iters)
        cfg.param_scheduler = [
            dict(
                type='LinearLR',
                start_factor=1e-3,
                by_epoch=False,
                begin=0,
                end=max(1, args.max_iters))
        ]
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.test_cfg = None
        cfg.test_dataloader = None
        cfg.test_evaluator = None
        cfg.default_hooks.checkpoint = dict(
            type='CheckpointHook',
            by_epoch=False,
            interval=args.max_iters)
    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
