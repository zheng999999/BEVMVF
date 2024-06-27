import torch
from mmdet3d.models import FUSION_LAYERS
from mmcv.runner import BaseModule
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer



class ChannelAttention(nn.Module):
    def __init__(self, inchannel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(inchannel, inchannel // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(inchannel // 16, inchannel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Block_stage1(nn.Module):

    def __init__(self, inchannel):
        super(Block_stage1, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, inchannel,kernel_size=3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(inchannel)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(inchannel)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()


    def forward(self,input:list):

        x = input[0]
        x_from_ca = input[1]
        x_from_li = input[2]
        residual = x
        out = self.ca(x) * x
        out = self.sa1(x_from_ca) * out
        out = self.sa2(x_from_li) * out
        out += residual
        out = self.relu(out)
        output = [out,x_from_ca,x_from_li]
        return output

class Block_stage2(nn.Module):

    def __init__(self, inchannel):
        super(Block_stage2, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, inchannel,kernel_size=3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(inchannel)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(inchannel)
        self.sa = SpatialAttention()


    def forward(self, input:list):
        x = input[0]
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        output = [out]
        return output


@FUSION_LAYERS.register_module()
class MyFusionlayer(BaseModule):
    """sampal concate the camera' bev and lidar's bev"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 layer_nums,
                 att_blocks_nums,
                 layer_strides,
                 norm_cfg,
                 conv_cfg,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(MyFusionlayer, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        in_filters = [in_channels, *out_channels[:-1]]
        ###
        self.cov1 = nn.Conv2d(64, 64,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        ###shengwei
        self.cov2 = nn.Conv2d(64, 128,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        ##att block part 

        self.att_block_stage1 = Block_stage1(in_channels)
        self.layer_1 = self._make_layer(Block_stage1,in_channels,block_num=att_blocks_nums) 
        self.layer_2 = self._make_layer(Block_stage2,in_channels,block_num=att_blocks_nums)   
        
        blocks = []
        
        ##down sample backbone part
        for i, layer_num in enumerate(layer_nums):
            backbone_block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                backbone_block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                backbone_block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                backbone_block.append(nn.ReLU(inplace=True))

            backbone_block = nn.Sequential(*backbone_block)
            blocks.append(backbone_block)
        self.blocks = nn.ModuleList(blocks)

    def _make_layer(self, block, channel, block_num):
    
        layers = []
        
        for i in range(block_num):
            layers.append(block(channel))

        return nn.Sequential(*layers)


    def forward(self, x_from_c ,x_from_l):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        
        x_from_l = self.cov1(x_from_l)
        x_from_l = self.bn1(x_from_l)
        x_from_l = self.relu(x_from_l)
        x = torch.cat((x_from_c,x_from_l),dim=1)
        ##
        x_from_c = self.cov2(x_from_c)
        x_from_c = self.bn2(x_from_c)
        x_from_c = self.relu(x_from_c)
        ##
        x_from_l = self.cov2(x_from_l)
        x_from_l = self.bn2(x_from_l)
        x_from_l = self.relu(x_from_l)
        input = [x,x_from_c,x_from_l]
        outs = []

        x = self.layer_1(input)
        x = self.layer_2(x)
        x = x[0]
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)

        
