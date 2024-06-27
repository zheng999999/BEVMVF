# -*- coding: utf-8 -*-
voxel_size = [0.16, 0.16, 4] ####
model = dict(
    type='Bev_Painting',
    style="v1",
    freeze_img=True,
    n_voxels=[432, 496, 25],
    voxel_size=[0.16, 0.16, 0.16],
    point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],

    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        frozen_stages = 3,
        init_cfg=dict(
            type='Pretrained', checkpoint='test_api_inference/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[320, 96, 32],
        out_channels=[96, 96, 96],
        init_cfg=dict(
            type='Pretrained', checkpoint='test_api_inference/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth')),

 
    neck_fuse=dict(in_channels=96*3, out_channels=64),
    change_channel = 16,
    neck_3d=dict(
        type='M2BevNeck_CutforPating',
        in_channels=64,
        out_channels=64,#######################
        num_layers=1,###########################
        stride=1,#######################
        norm_cfg=dict(type='BN', requires_grad=True)),
    
    pts_voxel_layer=dict(
        max_num_points=32,  # max_points_per_voxel
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)  # (training, testing) max_voxels
    ),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -1.78, 70.4, 39.68, -1.78],
            ],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# dataset settings
dataset_type = 'KittiDataset_Pating'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=True)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6))

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))
img_norm_cfg = dict(mean=[93.8329, 98.7603, 95.8774], std=[78.7817, 80.1303, 81.1997], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    #dict(type='MultiViewPipeline_2',n_images =)
    
    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KittiSetOrigin', point_cloud_range = point_cloud_range),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d','img'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),

    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KittiSetOrigin', point_cloud_range = point_cloud_range),
    dict(type='Collect3D', keys=['points', 'img'])
]
    
        
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    
    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KittiSetOrigin', point_cloud_range = point_cloud_range),
    dict(type='Collect3D', keys=['points', 'img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=2, pipeline=eval_pipeline)


optimizer = dict(type='AdamW', lr=0.000125, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# learning policy
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'

load_from = None #'pretrained_models/epoch_1.pth'
resume_from = None
workflow = [('train', 1)]

# fp16 settings, the loss scale is specifically tuned to avoid Nan
#fp16 = dict(loss_scale='dynamic')
