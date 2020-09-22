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

# @torchsnooper.snoop()
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a cnn3 network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders
    
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer([0,0])]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer([0,0])]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'Voc':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = VocDataset(parser.coco_path, set_name='2007',name = "train",
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer([460,640])]))
        dataset_val = VocDataset(parser.coco_path, set_name='2007',name = "trainval",
                                  transform=transforms.Compose([Normalizer(), Resizer([460,640])]))
                                  # transform=transforms.Compose([Normalizer(), Resizer([350,500])]),part = 1)
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)
    
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    
    # Create the model
    if parser.depth == 0:
        cnn3 = fractalnet.Fractalnet(num_classes=dataset_train.num_classes(), pretrained=True,istrain=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            cnn3 = cnn3.cuda()

    if torch.cuda.is_available():
        cnn3 = torch.nn.DataParallel(cnn3).cuda()
    else:
        cnn3 = torch.nn.DataParallel(cnn3)

    cnn3.training = True

    optimizer = optim.Adam(cnn3.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    
    cnn3.train()
    # cnn3.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    
    for epoch_num in range(parser.epochs):

        cnn3.train()
        # cnn3.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            # print(data['annot']['labels'])
            # print("@"*50)
            # continue
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    # classification_loss, regression_loss = cnn3([i for i in data['img'].cuda().float()], [{"boxes":i["boxes"],"labels":i["labels"]} for i in data['annot']])
                    print(cnn3([i for i in data['img'].cuda().float()], [{"boxes":i["boxes"],"labels":i["labels"]} for i in data['annot']]))
                    exit()
                else:
                    classification_loss, regression_loss = cnn3([i for i in data['img'].float()], [{"boxes":i["boxes"],"labels":i["labels"]} for i in data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(cnn3.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, cnn3)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, cnn3)

        scheduler.step(np.mean(epoch_loss))

        torch.save(cnn3.module, '{}_cnn3_460*640_{}.pt'.format(parser.dataset, epoch_num))

    cnn3.eval()

    torch.save(cnn3, 'high_model_final.pt')
    

if __name__ == '__main__':
    main()
