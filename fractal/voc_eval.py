from __future__ import print_function

import numpy as np
import json
import os

import torch
from prettytable import PrettyTable


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, cnn3, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the cnn3 using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the cnn3.
        cnn3           : The cnn3 to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    cnn3.eval()
    
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = cnn3(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = cnn3(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    cnn3,
    model_path,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
):
    """ Evaluate a given dataset using a given cnn3.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        cnn3           : The cnn3 to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """


    # gather all detections and annotations

    all_detections     = _get_detections(generator, cnn3, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)

    average_precisions = {}
    # if not os.path.isdir("Result"):
        # os.mkdir("Result")
    ALL_false_positives = np.zeros((0,))
    ALL_true_positives  = np.zeros((0,))
    three_scale = [[np.zeros((0,)),np.zeros((0,))],[np.zeros((0,)),np.zeros((0,))],[np.zeros((0,)),np.zeros((0,))]]
    three_num = [0,0,0]
    three_socre = [np.zeros((0,)),np.zeros((0,)),np.zeros((0,))]
    
    ALL_num = 0
    ALL_scores          = np.zeros((0,))
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        sma_false_positives = np.zeros((0,))
        sma_true_positives  = np.zeros((0,))
        mid_false_positives = np.zeros((0,))
        mid_true_positives  = np.zeros((0,))
        lar_false_positives = np.zeros((0,))
        lar_true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        sma_scores      = np.zeros((0,))
        mid_scores      = np.zeros((0,))
        lar_scores      = np.zeros((0,))
        num_annotations = 0.0
        sma_num = 0.0
        mid_num = 0.0
        lar_num = 0.0
        # f = open("Result\\" + str(label) + ".txt","w")
        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            ALL_num             += annotations.shape[0]
            detected_annotations = []
            
            for k in annotations:
                if((k[2] - k[0]) * (k[3] - k[1])) <= 1024:
                    sma_num += 1
                    three_num[0] += 1
                elif((k[2] - k[0]) * (k[3] - k[1]) <= 9216):
                    mid_num += 1
                    three_num[1] += 1
                else:
                    lar_num += 1
                    three_num[2] += 1
            for d in detections:
                scores = np.append(scores, d[4])
                ALL_scores = np.append(ALL_scores, d[4])
                # f.write(str(d) + "\n")
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    ALL_false_positives = np.append(ALL_false_positives, 1)
                    ALL_true_positives  = np.append(ALL_true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]
                # f.write("IOU:\n")
                # f.write(str(overlaps) + "\n")
                # f.write("Assigned_annotation:\n")
                # f.write(str(assigned_annotation) + "\n")
                target_annotations = annotations[assigned_annotation][0]
                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    ALL_false_positives = np.append(ALL_false_positives, 0)
                    ALL_true_positives  = np.append(ALL_true_positives, 1)
                    if(np.abs(target_annotations[2] - target_annotations[0]) * np.abs(target_annotations[3] - target_annotations[1]) <= 1024):
                        sma_false_positives = np.append(sma_false_positives, 0)
                        sma_true_positives  = np.append(sma_true_positives, 1)
                        sma_scores = np.append(sma_scores, d[4])
                        three_scale[0][0] = np.append(three_scale[0][0], 0)
                        three_scale[0][1] = np.append(three_scale[0][1], 1)
                        three_socre[0] = np.append(three_socre[0],d[4])
                    elif (np.abs(target_annotations[2] - target_annotations[0]) * np.abs(target_annotations[3] - target_annotations[1]) <= 9216):
                        mid_false_positives = np.append(mid_false_positives, 0)
                        mid_true_positives  = np.append(mid_true_positives, 1)
                        mid_scores = np.append(mid_scores, d[4])
                        three_scale[1][0] = np.append(three_scale[1][0], 0)
                        three_scale[1][1] = np.append(three_scale[1][1], 1)
                        three_socre[1] = np.append(three_socre[1],d[4])
                        # mid_num += 1
                    else:
                        lar_false_positives = np.append(lar_false_positives, 0)
                        lar_true_positives  = np.append(lar_true_positives, 1)
                        lar_scores = np.append(lar_scores, d[4])
                        three_scale[2][0] = np.append(three_scale[2][0], 0)
                        three_scale[2][1] = np.append(three_scale[2][1], 1)
                        three_socre[2] = np.append(three_socre[2],d[4])
                        # lar_num += 1
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    ALL_false_positives = np.append(ALL_false_positives, 1)
                    ALL_true_positives  = np.append(ALL_true_positives, 0)
                    if(np.abs(target_annotations[2] - target_annotations[0]) * np.abs(target_annotations[3] - target_annotations[1]) <= 1024):
                        sma_false_positives = np.append(sma_false_positives, 1)
                        sma_true_positives  = np.append(sma_true_positives, 0)
                        sma_scores = np.append(sma_scores, d[4])
                        three_scale[0][0] = np.append(three_scale[0][0], 1)
                        three_scale[0][1] = np.append(three_scale[0][1], 0)
                        three_socre[0] = np.append(three_socre[0],d[4])
                        # sma_num += 1
                    elif(np.abs(target_annotations[2] - target_annotations[0]) * np.abs(target_annotations[3] - target_annotations[1]) <= 9216):
                        mid_false_positives = np.append(mid_false_positives, 1)
                        mid_true_positives  = np.append(mid_true_positives, 0)
                        mid_scores = np.append(mid_scores, d[4])
                        three_scale[1][0] = np.append(three_scale[1][0], 1)
                        three_scale[1][1] = np.append(three_scale[1][1], 0)
                        three_socre[1] = np.append(three_socre[1],d[4])
                        # mid_num += 1
                    else:
                        lar_false_positives = np.append(lar_false_positives, 1)
                        lar_true_positives  = np.append(lar_true_positives, 0)
                        lar_scores = np.append(lar_scores, d[4])
                        three_scale[2][0] = np.append(three_scale[2][0], 1)
                        three_scale[2][1] = np.append(three_scale[2][1], 0)
                        three_socre[2] = np.append(three_socre[2],d[4])
                        # lar_num += 1

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0, 0, 0, 0, 0, 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]
        sma_indices         = np.argsort(-sma_scores)
        mid_indices         = np.argsort(-mid_scores)
        lar_indices         = np.argsort(-lar_scores)
        sma_false_positives = sma_false_positives[sma_indices]
        sma_true_positives  = sma_true_positives[sma_indices]
        mid_false_positives = mid_false_positives[mid_indices]
        mid_true_positives  = mid_true_positives[mid_indices]
        lar_false_positives = lar_false_positives[lar_indices]
        lar_true_positives  = lar_true_positives[lar_indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)
        sma_false_positives = np.cumsum(sma_false_positives)
        sma_true_positives  = np.cumsum(sma_true_positives)
        mid_false_positives = np.cumsum(mid_false_positives)
        mid_true_positives  = np.cumsum(mid_true_positives)
        lar_false_positives = np.cumsum(lar_false_positives)
        lar_true_positives  = np.cumsum(lar_true_positives)

        # compute recall and precision
        recall        = true_positives     / num_annotations
        precision     = true_positives     / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        sma_recall    = sma_true_positives / sma_num
        sma_precision = sma_true_positives / np.maximum(sma_true_positives + sma_false_positives, np.finfo(np.float64).eps)
        mid_recall    = mid_true_positives / mid_num
        mid_precision = mid_true_positives / np.maximum(mid_true_positives + mid_false_positives, np.finfo(np.float64).eps)
        lar_recall    = lar_true_positives / lar_num
        lar_precision = lar_true_positives / np.maximum(lar_true_positives + lar_false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        sma_average_precision  = _compute_ap(sma_recall, sma_precision)
        mid_average_precision  = _compute_ap(mid_recall, mid_precision)
        lar_average_precision  = _compute_ap(lar_recall, lar_precision)
        average_precisions[label] = average_precision, num_annotations, sma_average_precision, sma_num,mid_average_precision, mid_num,lar_average_precision,lar_num
    

    ALL_indices = np.argsort(-ALL_scores)
    ALL_false_positives = ALL_false_positives[ALL_indices]
    ALL_true_positives  = ALL_true_positives[ALL_indices]
    ALL_false_positives = np.cumsum(ALL_false_positives)
    ALL_true_positives  = np.cumsum(ALL_true_positives)
    ALL_recall        = ALL_true_positives     / ALL_num
    ALL_precision     = ALL_true_positives     / np.maximum(ALL_true_positives + ALL_false_positives, np.finfo(np.float64).eps) 
    ALL_average_precision  = _compute_ap(ALL_recall, ALL_precision)
    three_pre = []
    three_recal = []
    three_ap = []
    for i in range(3):
        three_indices = np.argsort(-three_socre[i])
        three_scale[i][0] = three_scale[i][0][three_indices]
        three_scale[i][1]  = three_scale[i][1][three_indices]
        three_scale[i][0] = np.cumsum(three_scale[i][0])
        three_scale[i][1]  = np.cumsum(three_scale[i][1])
        three_recal.append(three_scale[i][1]     / three_num[i])
        three_pre.append(three_scale[i][1]     / np.maximum(three_scale[i][1] + three_scale[i][0], np.finfo(np.float64).eps)) 
        three_ap.append(_compute_ap(three_recal[-1], three_pre[-1]))
    
    # print('\nmAP:',ALL_average_precision)
    table = PrettyTable(['categories','AP','APs','APm','APl'])
    # f = open(model_path + "val_result" + str(epoch_num) + ".txt","w")
    # f = open(model_path + ".txt","w")
    # for label in range(generator.num_classes()):
        # label_name = generator.label_to_name(label)
        # print('{} | \tAP:{} \t| \tAPs:{} \t| \tAPm:{} \t| \tAPl:{} \t|'.format(label_name, average_precisions[label][0], average_precisions[label][2], average_precisions[label][4], average_precisions[label][6]))
        # table.add_row([label_name, round(average_precisions[label][0],3), round(average_precisions[label][2],3), round(average_precisions[label][4],3), round(average_precisions[label][6],3)])
    
    table.add_row(["ALL:",round(ALL_average_precision,3)] + [round(i,3) for i in three_ap])
    # f.write(str(table))
    # f.close()
    print(table)
    return average_precisions

