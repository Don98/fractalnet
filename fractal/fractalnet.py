import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision.ops import nms
from fractal.utils import BasicBlock, BigBlock, Bottleneck, BBoxTransform, ClipBoxes
from fractal.anchors import Anchors
from fractal import losses
from torchvision.models.detection import FasterRCNN
import torchsnooper
from fractal.load_dic import load_dict
# @torchsnooper.snoop()
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


def Fractalnet(num_classes, pretrained=False, istrain = True,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # backbone = FractalNet(num_classes, BasicBlock, BigBlock,istrain, **kwargs)
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    state_dict = load_dict()
    torch.save(backbone.state_dict(), "backbone.pth")
    backbone.load_state_dict(state_dict,strict=False)
    backbone.out_channels = 1280
    model = FasterRCNN(backbone,num_classes=num_classes)
    return model

