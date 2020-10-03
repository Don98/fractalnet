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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
assert torch.__version__.split('.')[0] == '1'
import torchsnooper
print('CUDA available: {}'.format(torch.cuda.is_available()))

#@torchsnooper.snoop()
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a cnn3 network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

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
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    print("Num classes : ", dataset_train.num_classes())
    # Create the model

    cnn3 = fractalnet.Fractalnet(num_classes=dataset_train.num_classes(), pretrained=True,istrain=True)
    
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            cnn3 = cnn3.cuda()

    #if torch.cuda.is_available():
    #    cnn3 = torch.nn.DataParallel(cnn3,device_ids=[0]).cuda()
    #else:
    #    cnn3 = torch.nn.DataParallel(cnn3)

    cnn3.training = True

    optimizer = optim.Adam(cnn3.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    
    # cnn3.train()
    # cnn3.module.freeze_bn()
    cnn3 = cnn3.float()
    print('Num training images: {}'.format(len(dataset_train)))
    
    for epoch_num in range(parser.epochs):

        # cnn3.train()
        # cnn3.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            #images = [torch.tensor(i, dtype=torch.float).cuda() for i in data['img'].cuda().float()]
            images = []
            targets = []
            for i in range(len(data["annot"])):
                data["annot"][i]["labels"] = torch.tensor(data["annot"][i]["labels"],dtype=torch.int64)
                d = {}
                d["labels"] = data["annot"][i]["labels"].reshape((1,data["annot"][i]["labels"].shape[0]))[0].cuda()
                d["boxes"] = torch.tensor(data["annot"][i]["boxes"],dtype=torch.float).cuda()
                torch.save(d["labels"],"labels_"+str(i) + ".pt")
                torch.save(d["boxes"],"boxes_" + str(i) + ".pt")
                torch.save(data["img"][i],"img_"+str(i) + ".pt")
                #print("labels:",d["labels"])
                #print("-"*50)
                #print("boxes: ",d["boxes"])
                #print("-"*50)
                if d["boxes"].shape[0] != 0:
                    targets.append(d)
                    images.append(data['img'][i].float().cuda())
            output = cnn3(images, targets)
            #print(output)
            #print("="*50)
            loss_classifier  = output["loss_classifier"].cuda()
            loss_box_reg     = output["loss_box_reg"].cuda()
            loss_rpn_box_reg = output["loss_rpn_box_reg"].cuda()
            loss_objectness  = output["loss_objectness"].cuda()
            loss1 = loss_classifier + loss_box_reg
            loss2 = loss_rpn_box_reg + loss_objectness
            # loss.backward()
            loss_classifier.backward()
            loss_box_reg.backward()
            loss_rpn_box_reg.backward()
            loss_objectness.backward()
            
            torch.nn.utils.clip_grad_norm_(cnn3.parameters(), 0.1)

            optimizer.step()
            print("iter_num is : " , iter_num,"\tloss1 : " , loss1 , "\tloss2 : ",loss2)
        cnn3.eval()
        torch.save(cnn3, 'model'+str(epoch_num)+'.pt')
        cnn3.train()

        # print("="*50)
        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataloader_val, cnn3)
            # loss_hist.append(float(loss))
            # epoch_loss.append(float(loss))
                
if __name__ == '__main__':
    main()
