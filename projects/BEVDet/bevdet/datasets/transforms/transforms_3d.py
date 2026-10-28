# Copyright (c) Phigent Robotics. All rights reserved.
"""BEV data augmentation of BEVDet for mmdet3d 1.4."""
import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes


@TRANSFORMS.register_module()
class BEVAug(BaseTransform):
    """Bird-eye-view data augmentation.

    The rotation / scale / flip of the BEV space is recorded in the ``bda``
    matrix (3x4), which is applied by the view transformer, while the ground
    truth boxes are transformed accordingly.

    Args:
        bda_aug_conf (dict): Config of bda augmentation.
        classes (list[str]): Names of interest classes.
        is_train (bool): Whether to sample random augmentation parameters.
    """

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(
                    torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def transform(self, results):
        gt_boxes = results['gt_bboxes_3d'].tensor
        gt_boxes[:, 2] = gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]
        rotate_bda, scale_bda, flip_dx, flip_dy = \
            self.sample_bda_augmentation()
        bda_mat = np.zeros((3, 4), dtype=np.float32)
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda,
                                               scale_bda, flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot.numpy()
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['bda'] = bda_mat
        return results
