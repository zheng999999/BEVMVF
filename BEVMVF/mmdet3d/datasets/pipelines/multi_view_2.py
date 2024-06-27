import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose, RandomFlip, LoadImageFromFile
import ipdb


@PIPELINES.register_module()
class MultiViewPipeline_2:
    def __init__(self, transforms, n_images, n_times=2, sequential=False):
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.n_times = n_times
        self.sequential = sequential

    def __sort_list(self, old_list, order):
        new_list = []
        for i in order:
            new_list.append(old_list[i])
        return new_list

    def __call__(self, results):
        imgs = []
        extrinsics = []
        if not self.sequential:
            assert len(results['img_info']) == 6
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            ids_list = sorted(ids)  # sort & tolist
        else:
            assert len(results['img_info']) == 6 * self.n_times, f'img info: {len(results["img_info"])}, n_times: {self.n_times}'
            ids_list = np.arange(len(results['img_info'])).tolist()
        for i in ids_list:
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['lidar2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        # resort 2d box by random ids
        if 'gt_bboxes' in results.keys():
            gt_bboxes = self.__sort_list(results['gt_bboxes'], ids_list)
            gt_labels = self.__sort_list(results['gt_labels'], ids_list)
            gt_bboxes_ignore = self.__sort_list(results['gt_bboxes_ignore'], ids_list)
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels
            results['gt_bboxes_ignore'] = gt_bboxes_ignore

        results['lidar2img']['extrinsic'] = extrinsics
        return results



