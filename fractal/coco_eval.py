from pycocotools.cocoeval import COCOeval
import json
import torch
import numpy as np

def evaluate_coco(dataset, model, threshold=0.05):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            print(data["annot"])
            print(len(data["annot"]))
            exit()
            images = []
            targets = []
            
            for i in range(len(data["scale"])):
                data["annot"][i]["labels"] = torch.tensor(data["annot"][i]["labels"],dtype=torch.int64)
                d = {}
                d["labels"] = data["annot"][i]["labels"].reshape((1,data["annot"][i]["labels"].shape[0]))[0].cuda()
                d["boxes"] = torch.tensor(data["annot"][i]["boxes"],dtype=torch.float).cuda()
                if d["boxes"].shape[0] != 0:
                    targets.append(d)
                    images.append(data['img'][i].float().cuda())
            exit()
            
        for iter_num, data in enumerate(dataset):
            scale = data['scale']
            print(scale)
            images = []
            targets = []

            for i in range(len(data["annot"])):
                data["annot"][i]["labels"] = torch.tensor(data["annot"][i]["labels"],dtype=torch.int64)
                d = {}
                d["labels"] = data["annot"][i]["labels"].reshape((1,data["annot"][i]["labels"].shape[0]))[0].cuda()
                d["boxes"] = torch.tensor(data["annot"][i]["boxes"],dtype=torch.float).cuda()
                if d["boxes"].shape[0] != 0:
                    targets.append(d)
                    images.append(data['img'][i].float().cuda())
            if iter_num == 10:
                break
            prediction = model(images,targets)
            print(prediction)
            scores = []
            labels = []
            boxes  = []
            for i in prediction:
                scores.append(i["scores"])
                labels.append(i["labels"])
                boxes.append(i["boxes"])
            # correct boxes for image scale
            boxes /= scale
            print(boxes)
            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')
        print(len(results))
        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        model.train()

        return
