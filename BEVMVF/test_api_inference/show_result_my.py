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

def inference_multi_modality_detector(model, pcd, image, ann_file):
    """Inference point cloud with the multi-modality detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file[0])
    image_idx = int(re.findall(r'\d+', image)[-1])  # xxx/sunrgbd_000017.jpg
    print(image_idx)
    for x in data_infos:
        if int(x['image']['image_idx']) != image_idx:
            continue
        info = x
        break
    data = dict(
        pts_filename=pcd,
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[],
        lidar2img = dict())
    #data = test_pipeline(data)

    # TODO: this code is dataset-specific. Move lidar2img and
    #       depth2img to .pkl annotations in the future.
    # LiDAR to image conversion
    if box_mode_3d == Box3DMode.LIDAR:
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        lidar2img = P2 @ rect @ Trv2c
        data['lidar2img']['extrinsic'] = lidar2img
        data['lidar2img']['intrinsic'] = np.eye(4, dtype=np.float32)

        info_result = get_ann_info(info)
        data['gt_bboxes_3d'] = info_result['gt_bboxes_3d']
        data['bbox3d_fields'].append('gt_bboxes_3d')
        data['gt_labels_3d'] = info_result['gt_labels_3d']
        
        data = test_pipeline(data)
    # Depth to image conversion
    elif box_mode_3d == Box3DMode.DEPTH:
        rt_mat = info['calib']['Rt']
        # follow Coord3DMode.convert_point
        rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                           ]) @ rt_mat.transpose(1, 0)
        depth2img = info['calib']['K'] @ rt_mat
        data['img_metas'][0].data['depth2img'] = depth2img

    ##data = collate([data], samples_per_gpu=1)
    
        # this is a workaround to avoid the bug of MMDataParallel
    data['img_metas'] = [data['img_metas'].data]
    data['points'] = [data['points'].data.cuda(0)]
    data['img'] = data['img'].data.cuda(0)

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def show_det_result_meshlab(data,
                            result,
                            out_dir,
                            score_thr=0.0,
                            show=False,
                            snapshot=False):
    """Show 3D detection result by meshlab."""
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

    return file_name


def get_ann_info(data_infos):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = data_infos
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            Box3DMode.LIDAR, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox']

        selected = drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        CLASSES = ('car', 'pedestrian', 'cyclist')
        gt_labels = []
        for cat in gt_names:
            if cat in CLASSES:
                gt_labels.append(CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results
def remove_dontcare(ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'DontCare'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations
def drop_arrays_by_name(gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

if __name__ =='__main__':

    # test only one
    config = "configs/bevpainting/bevpainting_dataAug.py"
    check_point = "work_dir/bevpainting_dataAug_add_bran_40puls/epoch_2.pth"

    
    ann_file= 'data/kitti/kitti_infos_val.pkl',
    
    model = init_model(config,check_point)
    imgroot = "data/kitti/training/image_2/"
    ptsroot = "data/kitti/training/velodyne/"
    filename = [imgroot+"000754.png",imgroot+"001042.png",imgroot+"001336.png"]
    pts_filename = [ptsroot+"000754.bin",ptsroot+"001042.bin",ptsroot+"001336.bin"]
    
    for img_path,pts_path in zip(filename,pts_filename):

        point = np.fromfile(pts_path, dtype=np.float32)
        point = point.reshape(-1, 4)
        out_dir = "show_result/bevpainting"
        out_filename = pts_path.split('/')[-1].split('.')[-2]
        
        result,data = inference_multi_modality_detector(model,pts_path,img_path,ann_file)
        
        gt_bboxes = data['gt_bboxes_3d'].data.tensor.numpy()
        show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                            Box3DMode.DEPTH)
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
        inds = pred_scores > 0.3
        pred_bboxes = pred_bboxes[inds]
        show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                Box3DMode.DEPTH)
        
        points = data['points'][0].cpu().numpy()
        points = Box3DMode.convert(points, Box3DMode.LIDAR,
                                                Box3DMode.DEPTH)
        #points = points[..., [1, 0, 2]]
        #points[..., 0] *= -1
        show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        out_filename, show=True)

        print("finsh the ")