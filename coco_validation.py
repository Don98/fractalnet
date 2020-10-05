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
assert torch.__version__.split('.')[0] == '1'
import torchsnooper
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    cnn3 = fractalnet.Fractalnet(num_classes=dataset_val.num_classes(), pretrained=True,istrain=True)
    
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            cnn3 = cnn3.cuda()

    if torch.cuda.is_available():
        cnn3.load_state_dict(torch.load(parser.model_path))
        # retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        cnn3.load_state_dict(torch.load(parser.model_path))
        # retinanet = torch.nn.DataParallel(retinanet)

    cnn3.training = False
    cnn3.eval()
    # cnn3.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, cnn3)


if __name__ == '__main__':
    main()
