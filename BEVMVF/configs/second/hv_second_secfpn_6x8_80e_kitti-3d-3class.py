_base_ = [
    '../_base_/models/hv_second_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

resume_from = 'work_dir/second/hv_second_secfpn/epoch_27.pth'