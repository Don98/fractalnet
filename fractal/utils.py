import torch
import torch.nn as nn
import numpy as np

# import torchsnooper
# @torchsnooper.snoop()

def conv3x3(in_planes, out_planes, stride=1,kernel_size=3,padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,kernel_size = 3,padding=1,drop_ratio=0.3):
        print("basic inplanes: ",inplanes)
        print("basic planes: ",planes)
        print("-"*50)
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, kernel_size, padding)
        self.drop1 = nn.Dropout(drop_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.relu(out)
        out = self.bn1(out)
        
        return out


class BigBlock(nn.Module):

    def __init__(self, inplanes, planes, stride = 1, kernel_size = 3, padding=1, drop_ratio=0.3,last_one = False):
        super(BigBlock, self).__init__()
        # print("inplanes: ",inplanes)
        # print("planes: ",planes)
        # print("="*50)
        #第一列
        self.con0_0 = BasicBlock(inplanes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        
        #最上面一块        
        self.con2_0 = BasicBlock(inplanes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con3_0 = BasicBlock(inplanes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con3_1 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        
        #第二块
        self.con2_1 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con3_2 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con3_3 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con1_1 = BasicBlock(inplanes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
                
        #第三块
        self.con2_2 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con3_4 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con3_5 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
      
        #第四块
        self.con2_3 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con3_6 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
        self.con3_7 = BasicBlock(planes, planes, stride, kernel_size = 3, padding = padding , drop_ratio = drop_ratio)
      
        self.maxpool  = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=7, padding=1)
        
    def forward(self, x):
        residual = x
        #第一列
        x_con0_0 = self.con0_0(x)
        
        #最上面一块    
        x_con2_0 = self.con2_0(x)
        x_con3_0 = self.con3_0(x)
        x_con3_1 = self.con3_1(x_con3_0)
        
        x_con3_1_plus = (x_con3_1 + x_con2_0) / 2
        
        #第二块    
        x_con2_1 = self.con2_1(x_con3_1_plus)
        x_con3_2 = self.con3_2(x_con3_1_plus)
        x_con3_3 = self.con3_3(x_con3_2)
        x_con1_0 = self.con1_0(x)
        
        x_con3_3_plus = (x_con3_3 + x_con2_1 + x_con1_0) / 3
        
        #第三块    
        x_con2_2 = self.con2_2(x_con3_3_plus)
        x_con3_4 = self.con3_4(x_con3_3_plus)
        x_con3_5 = self.con3_5(x_con3_4)
        
        x_con3_5_plus = (x_con3_5 + x_con2_2) / 2
        
        #第四块    
        x_con2_3 = self.con2_3(x_con3_5_plus)
        x_con3_6 = self.con3_6(x_con3_5_plus)
        x_con3_7 = self.con3_7(x_con3_6)
        
        if last_one:
            x_con3_31_plus = (x_con0_0 + x_con1_1 + x_con2_3 + x_con3_7) / 4
            return self.maxpool1(x_con3_31_plus)
        else:
            pool0_0 = self.maxpool(x_con0_0)
            pool1_1 = self.maxpool(x_con1_1)
            pool2_3 = self.maxpool(x_con2_3)
            pool3_7 = self.maxpool(x_con3_7)
            return (pool0_0 + pool1_1 + pool2_3 + pool3_7) / 4

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
