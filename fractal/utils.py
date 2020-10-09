import torch
import torch.nn as nn
import numpy as np

import torchsnooper
#@torchsnooper.snoop()

def conv3x3(in_planes, out_planes, stride=1,kernel_size=3,padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=True)

def drop_path(feature,drop_ratio,nums):
    num = 0
    rand_num = torch.rand((nums))
    while(num):
        rand_num = torch.rand((nums))
        num = torch.sum(rand_num)
    rand_num = rand_num > drop_ratio
    new_nums = torch.sum(rand_num)
    new_feature = torch.zeros(feature[0].shape).cuda()
    for i in range(nums):
        if rand_num[i]:
            new_feature += feature[0]
    return new_feature / new_nums

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,kernel_size = 3,padding=1,drop_ratio=0.3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, kernel_size, padding)
        self.drop1 = nn.Dropout(drop_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        del x
        torch.cuda.empty_cache()
        out = self.drop1(out)
        out = self.relu(out)
        out = self.bn1(out)
        return out


class BigBlock(nn.Module):

    def __init__(self, inplanes, planes, stride = 1, kernel_size = 3, padding=1, drop_ratio=0.3,last_one = False):
        super(BigBlock, self).__init__()
        self.last_one = last_one
        self.drop_ratio = drop_ratio
        #��һ��
        self.conv0_0 = BasicBlock(inplanes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        
        #������һ��        
        self.conv2_0 = BasicBlock(inplanes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv3_0 = BasicBlock(inplanes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv3_1 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        
        #�ڶ���
        self.conv2_1 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv3_2 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv3_3 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv1_0 = BasicBlock(inplanes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
                
        #������
        self.conv2_2 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv3_4 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv3_5 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv1_1 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
      
        #���Ŀ�
        self.conv2_3 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv3_6 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.conv3_7 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
      
        self.maxpool  = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=7, padding=1)
        
    def forward(self, x):
        residual = x
        #��һ��
        x_con0_0 = self.conv0_0(x)
        
        #������һ��    
        x_con2_0 = self.conv2_0(x)
        x_con3_0 = self.conv3_0(x)
        x_con3_1 = self.conv3_1(x_con3_0)
        
        x_con1_0 = self.conv1_0(x)
        
        # x_con3_1_plus = (x_con3_1 + x_con2_0) / 2
        x_con3_1_plus = drop_path([x_con3_1,x_con2_0],self.drop_ratio,2)
        del x;del x_con2_0;del x_con3_0;del x_con3_1
        torch.cuda.empty_cache()
        #�ڶ���    
        x_con2_1 = self.conv2_1(x_con3_1_plus)
        x_con3_2 = self.conv3_2(x_con3_1_plus)
        x_con3_3 = self.conv3_3(x_con3_2)
        # x_con1_0 = self.conv1_0(x)
        
        # x_con3_3_plus = (x_con3_3 + x_con2_1 + x_con1_0) / 3
        x_con3_3_plus = drop_path([x_con3_3 , x_con2_1 , x_con1_0],self.drop_ratio,3)
        del x_con1_0;del x_con2_1;del x_con3_2;del x_con3_3
        torch.cuda.empty_cache()
        
        
        #������    
        x_con1_1 = self.conv1_1(x_con3_3_plus)
        x_con2_2 = self.conv2_2(x_con3_3_plus)
        x_con3_4 = self.conv3_4(x_con3_3_plus)
        x_con3_5 = self.conv3_5(x_con3_4)
        
        # x_con3_5_plus = (x_con3_5 + x_con2_2) / 2
        x_con3_5_plus = drop_path([x_con3_5 , x_con2_2],self.drop_ratio,2)
        del x_con3_3_plus;del x_con2_2;del x_con3_4;del x_con3_5
        torch.cuda.empty_cache()
        
        
        #���Ŀ�    
        x_con2_3 = self.conv2_3(x_con3_5_plus)
        x_con3_6 = self.conv3_6(x_con3_5_plus)
        x_con3_7 = self.conv3_7(x_con3_6)
        
        if self.last_one:
            x_con3_31_plus = drop_path([x_con0_0 , x_con1_1 , x_con2_3 , x_con3_7],self.drop_ratio,4)
            # x_con3_31_plus = (x_con0_0 + x_con1_1 + x_con2_3 + x_con3_7) / 4
            del x_con3_6;del x_con2_3;del x_con3_7
            torch.cuda.empty_cache()
            return self.maxpool1(x_con3_31_plus)
        else:
            pool0_0 = self.maxpool(x_con0_0)
            pool1_1 = self.maxpool(x_con1_1)
            pool2_3 = self.maxpool(x_con2_3)
            pool3_7 = self.maxpool(x_con3_7)
            x_con3_31_plus = drop_path([x_con0_0 , x_con1_1 , x_con2_3 , x_con3_7],self.drop_ratio,4)
            del x_con3_6;del x_con2_3;del x_con3_7
            torch.cuda.empty_cache()
            return x_con3_31_plus

        # return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes
