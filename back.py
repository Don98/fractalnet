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

#@torchsnooper.snoop()
class FractalNet(nn.Module):

    def __init__(self, num_classes, block, bigblock,istrain=True):
        self.inplanes = 64
        self.drop_ratio = 0
        self.training = istrain
        if self.training:
            self.drop_ratio = 0.3
        super(FractalNet, self).__init__()
        self.convH_0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.convH_0.bias.data = torch.zeros((64))
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
        x = self.drop1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpoolH_0(x)
        
        x1 = self.the_block1(x)
        x2 = self.the_block2(x1)
        x3 = self.the_block3(x2)
        x4 = self.the_block4(x3)
        
        # print("target ", annotations)
        # print("target size : ", annotations.shape)
        # print("x4 size : ", x4.shape)
        # print("="*50)
        return x4

def main(args=None):

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
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    backbone = FractalNet(80, BasicBlock, BigBlock,True)
    # images = []
    # targets = []
    # for i in range(2):
        # d = {}
        # d['boxes'] = torch.tensor(torch.load("boxes"+str(i) + ".pt"),dtype=torch.float).cuda()
        # d["labels"] = torch.load("labels" + str(i) + ".pt")

        # d["labels"] = d["labels"].reshape((1,d["labels"].shape[0]))[0].cuda()
        # targets.append(d)
        # images.append(torch.tensor(torch.load("img_" + str(i) + ".pt"),dtype=torch.float).cuda())
    model = FasterRCNN(backbone,num_classes=80)
    model = model.cuda()

    #model.training = True
    #optimizer = optim.Adam(model.parameters(), lr=1e-5)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    #loss_hist = collections.deque(maxlen=500)
    #output = model(images, targets)

    model = model.float()
    model.eval()
    coco_eval.evaluate_coco(dataloader_val, model)

    print(output) 

if __name__ == "__main__":
    main()