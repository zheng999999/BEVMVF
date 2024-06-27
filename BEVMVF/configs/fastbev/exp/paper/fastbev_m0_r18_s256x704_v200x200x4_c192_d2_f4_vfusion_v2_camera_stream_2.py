# -*- coding: utf-8 -*-
voxel_size = [0.25,0.25,8] ####
point_cloud_range = [-50, -50, -5, 50, 50, 3] ####

model = dict(
    type='FastBEV_CAM_V1',####
    style="v1",
    backbone=dict(
        type='ResNet',
        depth=18,###
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='pretrained_models/cascade_mask_rcnn_r18_fpn/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'),
        style='pytorch'
    ),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01, requires_grad=True),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=[256], out_channels=[64]),#####
    neck_3d=dict(
        type='M2BevNeck_V2',
        in_channels=64*4,
        out_channels=64,###
        num_layers=1,###
        stride=1,  ###
        is_transpose=False,
        fuse=dict(in_channels=64*4, out_channels=64*4),
        norm_cfg=dict(type='BN', requires_grad=True)),
    
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        layer_nums=[1, 1, 1],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),


    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1,2,4],
            sizes=[
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),


    ##################
    multi_scale_id=[0],
    n_voxels=[[400, 400, 4]],
    voxel_size=[[0.25, 0.25, 4]], #v2gengxin
    freeze_img=False,  ##v2 xinzeng
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=3000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500)
)


# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'NuScenesMultiView_Map_Dataset3'
data_root = './data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_config = {
    'src_size': (900, 1600),
    'input_size': (256, 704),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': (256, 704),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

file_client_args = dict(backend='disk')
""" file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        data_root: 'public-1984:s3://openmmlab/datasets/detection3d/nuscenes/'})) """

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=False, n_images=6, n_times=4, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D',
         with_bbox=True,
         with_label=True,
         with_bev_seg=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),

    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img','points', 'gt_bboxes', 'gt_labels',
                                 'gt_bboxes_3d', 'gt_labels_3d',
                                 'gt_bev_seg'])]
test_pipeline = [
    dict(type='MultiViewPipeline', sequential=False, n_images=6, n_times=4, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),

    # dict(type='TestTimeAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img','points'])]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        with_box2d=True,
        box_type_3d='LiDAR',
        ann_file='data/nuscenes/nuscenes_infos_train.pkl',
        load_interval=1,
        sequential=False,  ###
        n_times=4,
        train_adj_ids=[1, 3, 5],
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        prev_only=False,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_box2d=True,
        box_type_3d='LiDAR',
        ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        load_interval=1,
        sequential=False, ###
        n_times=4,
        train_adj_ids=[1, 3, 5],
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_box2d=True,
        box_type_3d='LiDAR',
        ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        load_interval=1,
        sequential=False, ###
        n_times=4,
        train_adj_ids=[1, 3, 5],
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None,
    )
)

###################
optimizer = dict(type='AdamW',
                 lr=0.00025, 
                 weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 100000,
    step=[20, 23])
momentum_config = None
##########################

total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
evaluation = dict(interval=20,pipeline=test_pipeline)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'

load_from = False
resume_from = None#'./work_dir/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4_vfusion_v2_camera_stream_2/epoch_6.pth'
workflow = [('train', 1)]

# fp16 settings, the loss scale is specifically tuned to avoid Nan
#fp16 = dict(loss_scale='dynamic')
