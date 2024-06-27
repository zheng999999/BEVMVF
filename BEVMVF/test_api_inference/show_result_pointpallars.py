# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import re
import torch
from copy import deepcopy
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from os import path as osp

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result, show_result,
                          show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.apis.inference import show_det_result_meshlab

def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def inference_detector(model, pcd):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data




""" def show_det_result_meshlab(data,
                            result,
                            out_dir,
                            score_thr=0.0,
                            show=False,
                            snapshot=False):
    
    points = data['points'][0].cpu().numpy()
    pts_filename = data['img_metas'][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    # for now we convert points into depth mode
    box_mode = data['img_metas'][0]['box_mode_3d']
    if box_mode != Box3DMode.DEPTH:
        points = points[..., [1, 0, 2]]
        points[..., 0] *= -1
        show_bboxes = Box3DMode.convert(pred_bboxes, box_mode, Box3DMode.DEPTH)
    else:
        show_bboxes = deepcopy(pred_bboxes)

    show_result(
        points,
        None,
        show_bboxes,
        out_dir,
        file_name,
        show=show,
        snapshot=snapshot)

    return file_name """

if __name__ =='__main__':

    # test only one
    config = "work_dir/hv_pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py"
    check_point = "work_dir/hv_pointpillars/epoch_48.pth"

    
    ann_file= 'data/kitti/kitti_infos_val.pkl',
    model = init_model(config,check_point)
    imgroot = "data/kitti/training/image_2/"
    ptsroot = "data/kitti/training/velodyne/"
    
    pts_filename = [ptsroot+"000754.bin",ptsroot+"001042.bin",ptsroot+"001336.bin"]
    for pts_path in pts_filename:

        point = np.fromfile(pts_path, dtype=np.float32)
        point = point.reshape(-1, 4)
        
        result,data = inference_detector(model,pts_path)

        out_dir = "show_result/pointpallars"
        out_filename = pts_path.split('/')[-1].split('.')[-2]
        
        show =show_det_result_meshlab(data,result,out_dir=out_dir,score_thr=0.1,show=True)
        print("finsh the {}".format(show))