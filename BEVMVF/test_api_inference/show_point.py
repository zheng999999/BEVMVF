
from mmdet3d.core.visualizer.open3d_vis import Visualizer

import numpy as np
# Load KITTI dataset
dataset_path = './data/kitti/training/velodyne/000274.bin'  # Replace with the actual path
point = np.fromfile(dataset_path,dtype=np.float32)
point = point.reshape(-1,4)

visualizer = Visualizer(point)

visualizer.show()


