# Copyright (c) Phigent Robotics. All rights reserved.
"""Transforms preparing multi-view images and calibrations for BEVDet."""
import os.path as osp

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PrepareImageInputs(BaseTransform):
    """Read multi-view images and prepare their calibration tensors.

    Unlike :class:`LoadMultiViewImageFromFiles`, this transform additionally
    generates the augmentation-related homographies that BEVDet needs, i.e.
    ``post_rot`` / ``post_trans``, which map coordinates of the augmented image
    back to those of the raw one.

    Args:
        data_config (dict): Data config of BEVDet, which must contain keys
            ``cams``, ``Ncams`` and ``input_size``.
        is_train (bool): Whether the training-time augmentation is applied.
    """

    def __init__(self,
                 data_config,
                 data_root=None,
                 is_train=False,
                 sequential=False):
        self.data_config = data_config
        self.data_root = data_root
        self.is_train = is_train
        self.sequential = sequential
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(self.data_config['cams'],
                                         self.data_config['Ncams'],
                                         replace=False)
        else:
            cam_names = self.data_config['cams']
        return list(cam_names)

    def get_rot(self, h):
        return torch.Tensor([[np.cos(h), np.sin(h)],
                             [-np.sin(h), np.cos(h)]])

    def sample_augmentation(self, H, W):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h']))
                         * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        return img, post_rot, post_tran

    def normalize_img(self, img):
        img = np.asarray(img).astype(np.float32)
        return (img - self.mean) / self.std

    def transform(self, results):
        imgs, sensor2egos, ego2globals, intrins = [], [], [], []
        post_rots, post_trans = [], []
        ego2global = np.asarray(results['ego2global'], dtype=np.float32)
        for cam_name in self.choose_cams():
            cam_data = results['images'][cam_name]
            img_path = cam_data['img_path']
            if not osp.exists(img_path):
                # some converted pkl files only keep the file name
                candidate = osp.join(self.data_root or '', 'samples', cam_name,
                                     osp.basename(img_path))
                if osp.exists(candidate):
                    img_path = candidate
                elif not osp.isabs(img_path) and self.data_root is not None:
                    img_path = osp.join(self.data_root, img_path)
            img = Image.open(img_path)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.Tensor(np.asarray(cam_data['cam2img']))

            # camera to ego(=lidar) transformation
            if 'cam2ego' in cam_data:
                sensor2ego = torch.Tensor(
                    np.asarray(cam_data['cam2ego'], dtype=np.float32))
            else:
                lidar2cam = np.asarray(
                    cam_data['lidar2cam'], dtype=np.float32)
                sensor2ego = torch.Tensor(np.linalg.inv(lidar2cam))

            # image view augmentation (resize, crop, flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                H=img.height, W=img.width)
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_tran, resize=resize,
                resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(torch.Tensor(ego2global))
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        results['img'] = imgs  # list of (H, W, 3) float32, packed by PACK
        results['sensor2ego'] = np.stack(
            [t.numpy() for t in sensor2egos]).astype(np.float32)
        results['ego2global'] = np.stack(
            [t.numpy() for t in ego2globals]).astype(np.float32)
        results['cam2img'] = np.stack([t.numpy()
                                       for t in intrins]).astype(np.float32)
        results['post_rot'] = np.stack(
            [t.numpy() for t in post_rots]).astype(np.float32)
        results['post_trans'] = np.stack(
            [t.numpy() for t in post_trans]).astype(np.float32)
        # identity bev augmentation, overridden by ``BEVAug`` at training time
        results['bda'] = np.concatenate(
            [np.eye(3), np.zeros((3, 1))], axis=1).astype(np.float32)
        return results
