import torch.nn as nn
import torch
import math
import collections
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from fractal.utils import BasicBlock, BigBlock, Bottleneck, BBoxTransform, ClipBoxes
from fractal.anchors import Anchors
from fractal import losses
from torchvision.models.detection import FasterRCNN
import torchsnooper
import torch.optim as optim
from fractal import coco_eval
from fractal import csv_eval

import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from fractal import model
from fractal import fractalnet
from fractal.dataloader import CocoDataset, CSVDataset, VocDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from fractal import coco_eval
from fractal import csv_eval
import os
import time

from torch.nn.parallel import DistributedDataParallel as DDP
#@torchsnooper.snoop()
class FractalNet(nn.Module):

    def __init__(self, num_classes, block, bigblock,istrain=True):
        self.inplanes = 64
        self.drop_ratio = 0
        self.training = istrain
        if self.training:
            self.drop_ratio = 0.3
        super(FractalNet, self).__init__()
        self.convH_0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.drop1 = nn.Dropout(self.drop_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpoolH_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.the_block1 = self._make_the_block(bigblock, inplanes = 64, planes = 128)
        self.the_block2 = self._make_the_block(bigblock, inplanes = 128, planes = 256)
        self.the_block3 = self._make_the_block(bigblock, inplanes = 256, planes = 512)
        self.the_block4 = self._make_the_block(bigblock, inplanes = 512, planes = 1024,last_one=True)
        self.out_channels = 1024
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze_bn()

    def _make_the_block(self, bigblock, inplanes, planes, last_one=False):
        layers = [bigblock(inplanes, planes, stride = 1, kernel_size = 3, padding=1, drop_ratio=0.3,last_one = last_one)]
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        img_batch = inputs
        x = self.convH_0(img_batch)
        del img_batch
        torch.cuda.empty_cache()
        x = self.drop1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpoolH_0(x)


        x1 = self.the_block1(x)
        del x
        torch.cuda.empty_cache()
        x2 = self.the_block2(x1)
        del x1
        torch.cuda.empty_cache()
        x3 = self.the_block3(x2)
        del x2
        torch.cuda.empty_cache()
        x4 = self.the_block4(x3)
        del x3
        torch.cuda.empty_cache()
        return x4

def get_layer_param(model):
    return sum([torch.numel(param) for param in model.parameters()])

def main(args=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    
    parser = argparse.ArgumentParser(description='Simple training script for training a cnn3 network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    
    parser = parser.parse_args(args)
    if parser.dataset == 'coco':
        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                        transform=transforms.Compose([Normalizer(), Augmenter(), Resizer([0,0])]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                      transform=transforms.Compose([Normalizer(), Resizer([0,0])]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)
                    
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    backbone = FractalNet(80, BasicBlock, BigBlock,True)
    model = FasterRCNN(backbone,num_classes=80)
    model = model.cuda()
    model = model.float()
    '''
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)
    images = [torch.ones(3,255,255,dtype=torch.float).cuda(),torch.zeros(3,255,255,dtype=torch.float).cuda()]
    targets = [{"boxes":torch.ones(1,4).cuda(),"labels":torch.ones(1,dtype=torch.int64).cuda()},{"boxes":torch.ones(1,4).cuda(),"labels":torch.ones(1,dtype=torch.int64).cuda()}]
    torch.cuda.empty_cache()
    for i in range(100):
        output = model(images,targets)
        print(output)
    exit()
    '''
    for epoch_num in range(1):

        model.train()
        # cnn3.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            #images = [torch.tensor(i, dtype=torch.float).cuda() for i in data['img'].cuda().float()]
            print(iter_num)
            images = []
            targets = []
            for i in range(len(data["annot"])):
                data["annot"][i]["labels"] = torch.tensor(data["annot"][i]["labels"],dtype=torch.int64)
                d = {}
                d["labels"] = data["annot"][i]["labels"].reshape((1,data["annot"][i]["labels"].shape[0]))[0].cuda()
                d["boxes"] = torch.tensor(data["annot"][i]["boxes"],dtype=torch.float).cuda()
                if d["boxes"].shape[0] != 0:
                    targets.append(d)
                    images.append(data['img'][i].cuda())
            if targets == []:
                continue
            
            if iter_num == 200:
                break
            output = model(images, targets)
            print(output)
            print("="*50)
            del images,targets
            torch.cuda.empty_cache()
            loss_classifier  = output["loss_classifier"].cuda()
            loss_box_reg     = output["loss_box_reg"].cuda()
            loss_rpn_box_reg = output["loss_rpn_box_reg"].cuda()
            loss_objectness  = output["loss_objectness"].cuda()
            loss1 = loss_classifier + loss_box_reg + loss_rpn_box_reg + loss_objectness
            
            print("loss1:\t",loss1)
            loss1.backward()
            del loss1
            torch.cuda.empty_cache()
    model.eval()
    coco_eval.evaluate_coco(dataset_val, model)

    print(output) 

if __name__ == "__main__":
    main()
