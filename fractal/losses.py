import numpy as np
import torch
import torch.nn as nn
import os

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        # print("Batch size : ",batch_size)
        # print("Class size : ",classifications.shape)
        # print("Ressg size : ",regressions.shape)
        # print(annotations)
        num = len(os.listdir("./"))
        f = open("record" + str(num) + ".txt","w")
        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            # print("bbox_annotation shape is : ",bbox_annotation.shape)
            # print(bbox_annotation)
            for i in range(bbox_annotation.shape[0]):
                f.write(str(bbox_annotation[i])[7:-18] + "\n")
            f.write("="*50 + "\n")

            if bbox_annotation.shape[0] == 0:
                # print(annotations)
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())
                f.write("0 0\n")
                continue
            
            if torch.cuda.is_available():
                each_bbox_loss = torch.zeros(bbox_annotation.shape[0]).cuda()
            else:
                each_bbox_loss = torch.zeros(bbox_annotation.shape[0])
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            # print("IOU shape is : " , IoU.shape)
            # print(IoU)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
            # print("IOU_max shape is : ",IoU_max.shape)
            # print(IoU_max)
            # print("IOU_argmax shape is : " ,IoU_argmax.shape)
            # print(IoU_argmax)
            
            #import pdb
            #pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            # print("Target shape is : ",targets.shape)
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)
            
            # print("positive_indices shape is ", positive_indices.shape)
            # print(positive_indices)
            num_positive_anchors = positive_indices.sum()
            # print(num_positive_anchors)
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            # print("assigned_annotations shape is : " ,assigned_annotations.shape)
            # print(assigned_annotations)
            # classP   True_CLASS
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            # print("target shape is : " ,targets.shape)
            # print(targets)
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha
            # = -(aplha)^gamma * log(classification) - (1 - alpha)^gamma * log(1 - classification) 
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # print("focal shape is ",focal_weight.shape)
            # print(focal_weight)
            # print("BCE shape is ",bce.shape)
            # print(bce)
            
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce
            # cls_loss = bce
            # print("CLS loss shape is : " , cls_loss.shape)
            # print(cls_loss)

            # print("cls_loss[0] : " ,cls_loss[0])
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            tmp1 = bce[positive_indices]
            # print("Tmp1 shape is : " ,tmp1.shape)
            # tmp5 = classification[positive_indices,assigned_annotations[positive_indices, 4].long()]
            tmp = classification[positive_indices,:]
            # print("Tmp shape is : " , tmp.shape)
            # P bbox_annotation clss_loss
            tmp2 = cls_loss[positive_indices]
            # print("Tmp2 shape is : " , tmp2.shape)
            
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            
            f.write(str(classification_losses) + "\n")    
            # print("clss_loss sum is : ",cls_loss.sum())
            # print("num_positive_anchors is : ",cls_loss.sum())
            # print("classification_losses shape is : " , len(classification_losses))
            # print(classification_losses)
            
            # print("final:")
            # print(alpha_factor[positive_indices][0])
            # print(focal_weight[positive_indices][0])
            # print(bce[positive_indices][0])
            # print(cls_loss[positive_indices][0])
            # compute the loss for regression
            # start = 0;end = 0;start1 = 0;end1 = 0
            # if tmp.shape[0] != 0:
                # start  = str(tmp1[0]).index("[")
                # end    = str(tmp1[0]).index("]")
                # start1 = str(tmp[0]).index("[")
                # end1   = str(tmp[0]).index("]")
            # else:
                # print(tmp.shape,positive_indices.sum())
                # print(classification.shape)
                # print(bce.shape)
                # print(cls_loss.shape)
                # print(cls_loss.sum())
                # print(torch.clamp(num_positive_anchors.float(), min=1.0))
            the_Iou_argmax = IoU_argmax[positive_indices]
            for i in range(tmp.shape[0]):
                # print('{}'.format(tmp[i].data))
                # length1 = len(str(tmp1[i]))
                # length  = len(str(tmp[i]))
                # if(start >=  length1 or str(tmp1[i])[start] != "["):
                    # start  = str(tmp1[i]).index("[")
                # if(end >= length1 or str(tmp1[i])[end] != "]"):
                    # end    = str(tmp1[i]).index("]")
                # if(start1 >= length or str(tmp[i])[start1] != "["):
                    # start1 = str(tmp[i]).index("[")
                # if(end1 >= length or str(tmp[i])[end1] != "]"):
                    # end1   = str(tmp[i]).index("]")
                f.write(str(tmp[i])+ " "+ str(the_Iou_argmax[i].item()) + " " + str(tmp1[i]) + " " + str(tmp2[i].sum().item()) + "\n") 
            f.write("-"*50+"\n")
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()
                # print("New Target shape is ",targets.shape)
                # print(targets)
                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)
                # print("~positive_indices shape is : ")
                # print(~positive_indices)
                # print("negative_indices shape is :" ,negative_indices.shape)
                # print(negative_indices)
                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                # print("regression_loss shape is : ", regression_loss.shape)
                # print(regression_loss)
                regression_losses.append(regression_loss.mean())
                f.write(str(regression_losses[-1].item()) + "\n")
                # start = str(regression_loss[0]).index("[")
                # end = str(regression_loss[0]).index("]")
                for i in range(regression_loss.shape[0]):
                    # if(str(regression_loss[i])[start] != "["):
                        # start = str(regression_loss[i]).index("[")
                    # if(str(regression_loss[i])[end] != "["):
                        # end = str(regression_loss[i]).index("]")
                    f.write(str(IoU_argmax[positive_indices][i].item()) + " " + str(regression_loss[i]) + " " + str(IoU_max[positive_indices][i].item()) + " " + str(anchor[positive_indices][i]) + "\n")
                
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                f.write("0")
        result = torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
        f.write(str((classification_losses[-1] + regression_losses[-1]).item()) + "\n")
        f.close()
        return result

    
