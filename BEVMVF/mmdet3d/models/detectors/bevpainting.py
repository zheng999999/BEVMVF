# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet3d.core import show_result

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet3d.models import build_voxel_encoder,build_middle_encoder,build_fusion_layer
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result

from mmcv.runner import get_dist_info, auto_fp16,force_fp32
from mmdet3d.ops import Voxelization
import copy
from mmcv.cnn import ConvModule
import mmcv
from mmdet.core.visualization import imshow_det_bboxes
import numpy as np
from mmdet3d.core.visualizer.image_vis import project_pts_on_img
import time

@DETECTORS.register_module()
class Bev_Painting(BaseDetector):
    def __init__(
        self,
        n_voxels=None,
        voxel_size=None,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        freeze_img=False,

        backbone=None,
        neck=None,
        neck_fuse=None,
        change_channel = 16,

        neck_3d=None,
    
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,

        pts_backbone=None,
        pts_neck=None,
        bbox_head=None,
        
        train_cfg=None,
        test_cfg=None,

        init_cfg=None,
        
        with_cp=False,
        style='v3',
    ):
        super().__init__(init_cfg=init_cfg)
        if backbone:
            self.backbone = build_backbone(backbone)

        if neck:
            self.neck = build_neck(neck)
        if neck_3d:
            self.neck_3d = build_neck(neck_3d)
    
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        
        self.freeze_img = freeze_img
        self.freeze()

        ########################
        #pts init
        ########################
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
            self.use_pts=True
        else:
            self.use_pts=False
        if pts_voxel_encoder:
            self.pts_voxel_encoder = build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = build_middle_encoder(
                pts_middle_encoder)

        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone)

        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)


        # style
        # v1: fastbev wo/ ms
        # v2: fastbev + img ms
        # v3: fastbev + bev ms
        # v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style


        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            #self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # checkpoint
        self.with_cp = with_cp
        #change channel

        
        self.change_channel_cov = nn.Conv2d((neck_fuse['out_channels']*n_voxels[2]), change_channel, kernel_size=7, stride=1,padding=3)
        
        #注意力融合模块用 调节输出通道
        self.fusion_attention = Fusion_Attention(change_channel+pts_middle_encoder['in_channels'])
        #残差融合模块用 调节输出通道
        self.change_channel_cov2 = nn.Conv2d((change_channel+2*pts_middle_encoder['in_channels']), pts_middle_encoder['in_channels'], kernel_size=7, stride=1,padding=3)
        
        #融合模块均不用 调节输出通道
        self.change_channel_cov3 = nn.Conv2d((change_channel+pts_middle_encoder['in_channels']), pts_middle_encoder['in_channels'], kernel_size=7, stride=1,padding=3)
        


    #freeze layer
    def freeze(self):
        if self.freeze_img:
            if self.backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            if self.neck:
                for param in self.neck.parameters():
                    param.requires_grad = False


    @staticmethod
    def _compute_projection(img_meta, stride):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"])
        intrinsic[:2] /= stride
        extrinsic = img_meta["lidar2img"]["extrinsic"]
        
        projection = (intrinsic @ extrinsic)
        return projection

    def extract_img_feat(self, img, img_metas, mode='train'):
        if len(img.shape)==3:
            img = torch.unsqueeze(img, dim=0) 
        batch_size = img.shape[0]  #img.shape = (2,375,1242,3)
        #img = img.permute(0,3,1,2)
        img = img.type(torch.float32)
        x = self.backbone(img)  

        # use for vovnet
        if isinstance(x, dict):
            tmp = []
            for k in x.keys():
                tmp.append(x[k])
            x = tmp

        # fuse features
        def _inner_forward(x):
            out = self.neck(x)
            return out  # [6, 64, 232, 400]; [6, 64, 116, 200]; [6, 64, 58, 100]; [6, 64, 29, 50])

        if self.with_cp and x.requires_grad:
            c1, c2, c3 = cp.checkpoint(_inner_forward, x)
        else:
            c1, c2, c3 = _inner_forward(x)


        c1 = F.interpolate(c1, size=c3.shape[2:], mode='bilinear', align_corners=False)
        c2 = F.interpolate(c2, size=c3.shape[2:], mode='bilinear', align_corners=False)

        

        x = torch.cat([c1, c2, c3], dim=1)

       
        def _inner_forward(x):
            out = self.neck_fuse(x)  
            return out
            
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        
        x = torch.unsqueeze(x,1) #(batch,1,c,h,w)

        stride = img.shape[-1] // x.shape[-1]  # 4.0
        #assert stride == 4
    

        # reconstruct 3d voxels
        volumes, valids = [], []
        
        for feature, img_meta in zip(x, img_metas):
            # feature: [6, 64, 232, 400]
            if isinstance(img_meta["img_shape"], list):
                img_meta["img_shape"] = img_meta["img_shape"][0]
            projection = self._compute_projection(img_meta, stride).to(x.device)  # [ 4, 4]

            points = get_points(  # [3, 200, 200, 12]
                n_voxels=torch.tensor(self.n_voxels,dtype=torch.float),
                voxel_size=torch.tensor(self.voxel_size,dtype=torch.float),
                #origin=torch.tensor([34.56,0,-1],dtype=torch.float),
                point_cloud_range = self.point_cloud_range
            ).to(x.device)

            height = img_meta["pad_shape"][0] // stride
            width = img_meta["pad_shape"][1] // stride

            if self.style == "v1":
                volume, valid = backproject(
                    feature[:, :, :height, :width], points, projection,img,img_metas
                )

                volume = volume.sum(dim=0)  # [6, 64, 200, 200, 12] -> [64, 200, 200, 12]
                valid = valid.sum(dim=0)  # [6, 1, 200, 200, 12] -> [1, 200, 200, 12]
                volume = volume / valid
                valid = valid > 0
                volume[:, ~valid[0]] = 0.0
            elif self.style == "v2":
                volume = backproject_v2(
                    feature[:, :, :height, :width], points, projection
                )  # [64, 200, 200, 12]
            else:
                volume = backproject_v3(
                    feature[:, :, :height, :width], points, projection
                )  # [64, 200, 200, 12]
            volumes.append(volume)

        x = torch.stack(volumes)  # [1, 64, 200, 200, 12]

        N, C, X, Y, Z = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N, X, Y, Z*C).permute(0, 3, 2, 1)
        x = self.change_channel_cov(x)
        return x
    
    def extract_pts_feat(self, points ):
        """Extract features of points."""
        
        x = self.pts_backbone(points)
      
        x = self.pts_neck(x)
        return x
    

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_feat(self, img, img_metas, points):
        """Extract features from images and points."""
        
        #point voxelize
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        pts_voxel = self.pts_middle_encoder(voxel_features, coors, batch_size)
        #img
        img2pts = self.extract_img_feat(img,img_metas)
        
        #fusion
        if img2pts.shape[2:] != pts_voxel.shape[2:]:
            self.img2pts = F.interpolate(img2pts, pts_voxel.shape[2:], mode='bilinear', align_corners=True)
        fusion_feats = torch.cat([img2pts,pts_voxel],dim=1)

        #add branch
        
        fusion_feats = self.fusion_attention(fusion_feats)

        fusion_feats = torch.cat((pts_voxel,fusion_feats),dim=1)
        fusion_feats = self.change_channel_cov2(fusion_feats)

        out = self.neck_3d(fusion_feats)
        #out = self.change_channel_cov3(fusion_feats)
        out = self.extract_pts_feat(out)

        
        return out
        
        
    
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    @auto_fp16(apply_to=('img','points' ))
    def forward(self, img, img_metas, points, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            if kwargs["export_2d"]:
                return self.onnx_export_2d(img, img_metas)
            elif kwargs["export_3d"]:
                return self.onnx_export_3d(img, img_metas)
            else:
                raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, points, **kwargs)
        else:
            return self.forward_test(img, img_metas, points, **kwargs)

    def forward_train(
        self, img, img_metas, points, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    ):
        out = self.extract_feat(img, img_metas, points)
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None

        
        
        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(out)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)
        return losses

    def forward_test(self, img, img_metas, points, **kwargs):
        if not self.test_cfg.get('use_tta', False):
            return self.simple_test(img, img_metas,points)
        return self.aug_test(img, img_metas,points)

    def onnx_export_2d(self, img, img_metas):
        """
        input: 6, 3, 544, 960
        output: 6, 64, 136, 240
        """
        x = self.backbone(img)
        c1, c2, c3= self.neck(x)
        c1 = F.interpolate(c1, size=c3.shape[2:], mode='bilinear', align_corners=False)
        c2 = F.interpolate(c2, size=c3.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([c1, c2, c3], dim=1)
        x = self.neck_fuse(x)

        if bool(os.getenv("DEPLOY", False)):
            x = x.permute(0, 2, 3, 1)
            return x

        return x

    def onnx_export_3d(self, x, _):
        # x: [6, 200, 100, 3, 256]
        # if bool(os.getenv("DEPLOY_DEBUG", False)):
        #     x = x.sum(dim=0, keepdim=True)
        #     return [x]
        if self.style == "v1":
            x = x.sum(dim=0, keepdim=True)  # [1, 200, 100, 3, 256]
            x = self.neck_3d(x)  # [[1, 256, 100, 50], ]
        elif self.style == "v2":
            x = self.neck_3d(x)  # [6, 256, 100, 50]
            x = [x[0].sum(dim=0, keepdim=True)]  # [1, 256, 100, 50]
        elif self.style == "v3":
            x = self.neck_3d(x)  # [1, 256, 100, 50]
        else:
            raise NotImplementedError

        if self.bbox_head is not None:
            cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
            cls_score = [item.sigmoid() for item in cls_score]

        if os.getenv("DEPLOY", False):
            if dir_cls_preds is None:
                x = [cls_score, bbox_pred]
            else:
                x = [cls_score, bbox_pred, dir_cls_preds]
            return x

        return x

    def simple_test(self, img, img_metas, points):
        bbox_results = []
        fusion_feature_bev = self.extract_feat(img, img_metas, points)
        if self.bbox_head is not None:
            x = self.bbox_head(fusion_feature_bev)
            bbox_list = [dict() for i in range(len(img_metas))]
            bbox_ = self.bbox_head.get_bboxes(*x, img_metas, valid=None,rescale=True)
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_
            ]
            for result_dict, pts_bbox in zip(bbox_list, bbox_results):
                result_dict['pts_bbox'] = pts_bbox

        else:
            bbox_list = [dict()]
        return bbox_list

    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):

            img_metas[0]['img_shape'] = img_shape_copy[24*tta_id:24*(tta_id+1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24*tta_id:24*(tta_id+1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_img_feat(imgs[:, 24*tta_id:24*(tta_id+1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    


@torch.no_grad()
def get_points(n_voxels, voxel_size, point_cloud_range): #n_voxels=[[220, 250, 1]], voxel_size = [0.16, 0.16, 0.5], origin
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    #new_origin = origin - n_voxels * 0.5 * voxel_size
    new_origin = point_cloud_range[:3]
    new_origin = torch.Tensor(new_origin)
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject(features, points, projection,img,img_meta):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape #[1,64,48,156]
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:] #[440,500,8]
    # [3, 440, 500, 8] -> [1, 3, 1760000] : [nimg, c , h*w*z]
    """ test_vis_point = points.view(3, -1).permute(1,0)
    test_vis_point = test_vis_point.cpu().numpy()
    test_img = img[0].cpu().numpy()
    test_projection = img_meta[0]['lidar2img']['extrinsic']
    project_pts_on_img(test_vis_point,test_img,test_projection)
    time.sleep(100) """
    points = points.view(1, 3, -1)
    # [1, 3, 1760000] -> [1, 4, 1760000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    projection = projection.view(1,4,4)
    # ego_to_cam
    # [1, 4, 4] * [1, 4, 1760000] -> [1, 4, 1760000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [1, 1760000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [1, 1760000]
    tem = points_2d_3[:, 0]
    z = points_2d_3[:, 2]  # [1, 1760000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height)  # [1, 1760000]  mask
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [1, 64, 1760000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [1, 64, 1760000] -> [1, 64, 440,500,8]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [1, 1760000] -> [1, 1, 440,500,8]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


def backproject_v2(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    # print(f"valid: {valid.shape}, percept: {valid.sum() / (valid.shape[0] * valid.shape[1])}")

    # method1：特征填充，只填充有效特征，重复特征加和平均
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    count = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] += features[i, :, y[i, valid[i]], x[i, valid[i]]]
        count[:, valid[i]] += 1
    volume[count > 0] /= count[count > 0]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume


def backproject_v3(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    # print(f"valid: {valid.shape}, percept: {valid.sum() / (valid.shape[0] * valid.shape[1])}")

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume


 
class Fusion_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.se = Change_SE_Block(dim//2)
        
 
    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = self.se(attn1,attn2)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
 
 
class Fusion_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
 
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = Fusion_block(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
 
    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
class Change_SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self,x1,x2):
        a = self.att2(x2)
        b = self.att1(x1)
        x1 = x1*a
        x2 = x2*b
        out = torch.cat((x1,x2),dim=1)
        return out