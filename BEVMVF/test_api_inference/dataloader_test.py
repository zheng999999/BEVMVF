# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
from mmdet.apis import set_random_seed
from mmcv import Config, DictAction

from os import path as osp

mmdet3d_root = os.environ.get('MMDET3D')
if mmdet3d_root is not None and osp.exists(mmdet3d_root):
    import sys
    sys.path.insert(0, mmdet3d_root)
    print(f"using mmdet3d: {mmdet3d_root}")

from mmdet3d.datasets import build_dataset,build_dataloader
from mmdet3d.models import build_model

from mmdet3d.datasets.kitti_dataset import KittiDataset
from torch.utils.data import DataLoader

cfg = Config.fromfile("./configs/bevpainting/bevpainting.py")
model = build_model(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))

datasets = build_dataset(cfg.data.train)
#dataset = datasets if isinstance(datasets, (list, tuple)) else [datasets]

seed = set_random_seed(0, deterministic=True)

data_loaders = [
        build_dataloader(
            datasets,
            1,
            1,
            # cfg.gpus will be ignored if distributed
            1,
            dist=False,
            seed=seed,
            shuffle=cfg.get('shuffle', True))
]

step = 0
data = []
for data in data_loaders[0]:
        img_metas = data['img_metas']
        print(type(img_metas))
        data.append(data)
        step = step + 1
        if step>5:
              break
print(data)
print("f")

    

