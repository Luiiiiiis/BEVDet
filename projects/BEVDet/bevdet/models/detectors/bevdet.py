# Copyright (c) Phigent Robotics. All rights reserved.
"""BEVDet detector migrated onto mmdet3d 1.4 (mmdet/mmcv 2.x) APIs."""
from typing import Dict, List, Optional

import numpy as np
import torch

from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class BEVDet(Base3DDetector):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    It keeps the original topology (image encoder -> view transformer -> BEV
    encoder -> detection head) while following the mmdet3d 1.4 detector
    protocol, i.e. ``loss(batch_inputs_dict, batch_data_samples)`` and
    ``predict(batch_inputs_dict, batch_data_samples)``.
    """

    def __init__(self,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 img_view_transformer: Optional[dict] = None,
                 img_bev_encoder_backbone: Optional[dict] = None,
                 img_bev_encoder_neck: Optional[dict] = None,
                 pts_bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor, **kwargs)
        self.img_backbone = MODELS.build(img_backbone)
        self.with_img_neck = img_neck is not None
        if self.with_img_neck:
            self.img_neck = MODELS.build(img_neck)
        self.img_view_transformer = MODELS.build(img_view_transformer)
        self.img_bev_encoder_backbone = MODELS.build(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = MODELS.build(img_bev_encoder_neck)
        self.with_pts_bbox_head = pts_bbox_head is not None
        if self.with_pts_bbox_head:
            head = dict(pts_bbox_head)
            if train_cfg is not None:
                head['train_cfg'] = train_cfg.get('pts', None)
            if test_cfg is not None:
                head['test_cfg'] = test_cfg.get('pts', None)
            self.pts_bbox_head = MODELS.build(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_bbox_head(self):
        return self.with_pts_bbox_head

    def image_encoder(self, img):
        """Encode multi-view images into an image-view feature tensor."""
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if isinstance(x, (list, tuple)):
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        """Convert gathered metainfo to the view-transformer argument."""
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins, post_rots,
                post_trans, bda]

    def extract_img_feat(self, img_inputs):
        """Extract the bird-eye-view feature from `img_inputs`."""
        inputs = self.prepare_inputs(img_inputs)
        x = self.image_encoder(inputs[0])
        x, depth = self.img_view_transformer([x] + inputs[1:7])
        x = self.bev_encoder(x)
        return [x], depth

    def build_img_inputs(
            self, batch_inputs_dict: dict,
            batch_data_samples: List[Det3DDataSample]) -> list:
        """Stack the per-sample metainfo into batched tensors."""
        imgs = batch_inputs_dict['imgs']  # (B, N, C, H, W)
        metas = [sample.metainfo for sample in batch_data_samples]

        def to_tensor(key):
            return imgs.new_tensor(
                np.stack([np.asarray(meta[key]) for meta in metas]))

        return [
            imgs, to_tensor('sensor2ego'), to_tensor('ego2global'),
            to_tensor('cam2img'), to_tensor('post_rot'),
            to_tensor('post_trans'), to_tensor('bda')
        ]

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: List[Det3DDataSample]) -> list:
        img_inputs = self.build_img_inputs(batch_inputs_dict,
                                           batch_data_samples)
        img_feats, _ = self.extract_img_feat(img_inputs)
        return img_feats

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: List[Det3DDataSample],
                 mode: str = 'tensor',
                 **kwargs):
        img_feats = self.extract_feat(batch_inputs_dict, batch_data_samples)
        if mode == 'tensor':
            return img_feats
        elif mode == 'loss':
            return self.pts_bbox_head.loss(img_feats, batch_data_samples,
                                           **kwargs)
        elif mode == 'predict':
            results_list = self.pts_bbox_head.predict(
                img_feats, batch_data_samples, **kwargs)
            return self.add_pred_to_datasample(batch_data_samples,
                                               results_list)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        return self._forward(batch_inputs_dict, batch_data_samples,
                             mode='loss', **kwargs)

    def predict(self, batch_inputs_dict: dict,
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        return self._forward(batch_inputs_dict, batch_data_samples,
                             mode='predict', **kwargs)
