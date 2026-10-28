# BEVDet-r50 migrated to mmdet3d 1.4 (mAP 28.3 / NDS 35.0 for the original)
default_scope = 'mmdet3d'

custom_imports = dict(imports=['bevdet'], allow_failed_imports=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5), scale_lim=(0.95, 1.05), flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64

input_modality = dict(use_lidar=False, use_camera=True)

data_root = '/home/dev/papers/BEVDet/data/nuscenes/'

meta_keys = ('sensor2ego', 'ego2global', 'cam2img', 'post_rot', 'post_trans',
             'bda', 'box_type_3d', 'box_mode_3d', 'sample_idx', 'num_pts_feats',
             'depth2img', 'lidar2cam', 'lidar2img', 'ori_cam2img',
             'num_ref_frames', 'num_views', 'scale_factor', 'flip',
             'flip_direction', 'token')

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=data_config,
        data_root=data_root,
        is_train=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=meta_keys),
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=data_config,
        data_root=data_root,
        is_train=False),
    dict(type='Pack3DDetInputs', keys=['img'], meta_keys=meta_keys),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR',
        use_valid_flag=True,
        test_mode=False))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=dict(classes=class_names),
        box_type_3d='LiDAR',
        use_valid_flag=True,
        test_mode=True))

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=None)

model = dict(
    type='BEVDet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=1),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2,
        start_level=0),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=class_names),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
        test_cfg=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,
            nms_type='rotate',
            nms_thr=0.2,
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55, 1.1, 1.0, 1.0, 1.5,
                                 3.5]])))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=5, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=2e-5,
        by_epoch=True,
        begin=0,
        end=20)
]

env_cfg = dict(cudnn_benchmark=True)
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=5)
val_cfg = dict()
test_cfg = dict()
