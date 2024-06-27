import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, 'image_2'))
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'image_2', self.images[idx])
        image = Image.open(img_path).convert('RGB')
        img =np.array(image)  # (w, h) ---> (h, w, c)
        img = img.transpose(2, 0, 1)  # (h, w, c) ---> (c, h, w)
        img = torch.tensor(img)
        img = img.to(torch.float32)

        if self.transform:
            image = self.transform(image)
        return img
 
# 计算数据集均值和标准差
def calculate_mean_std(dataset):
    mean = torch.zeros(3).to(torch.float32)
    std = torch.zeros(3).to(torch.float32)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for images in (data_loader):
        for i in range(3):
            mean[i] += images[:,i,:,:].mean()
            std[i] += images[:,i,:,:].std()
    
    mean.div_(len(dataset))
    std.div_(len(dataset))
    
    return mean, std
 
# 示例使用
root_dir = './tools/'

dataset = KITTIDataset(root_dir, transform=None)
 
# 计算KITTI数据集的均值和标准差
mean, std = calculate_mean_std(dataset)
print(f'Mean: {mean}, Standard Deviation: {std}')