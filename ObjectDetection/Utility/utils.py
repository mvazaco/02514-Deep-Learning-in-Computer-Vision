import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def no_max_supression(bboxes, iou_threshold=0.4) -> list[list[float]]:

    #bbox : [[x, y, w, h], class, confidence] 
    #bboxes : list[bbox]

    bbox_list_result = []
    
    #Sort and filter bbox
    boxes_sorted = sorted(bboxes, key=lambda x: x[2], reverse=True)
    
    #Remove boxes with high IOU
    while boxes_sorted:
        current_bbox = boxes_sorted.pop() #get bbox with highest confidence
        bbox_list_result.append(current_bbox)

        for bbox in boxes_sorted[:]:
            if current_bbox[1] == bbox[1]: #if same class
                iou = calculate_iou(current_bbox[0], bbox[0])
                if iou > iou_threshold:
                    boxes_sorted.remove(bbox)

    return bbox_list_result


#mAP
def mean_average_precision(bbox_true : list[list[float]], bbox_pred : list[list[float]], threshold_iot : float = 0.5, plot = False) -> float:

    boxes_sorted = sorted(bbox_pred, key=lambda x: x[2], reverse=True)

    #slice duplicate bbox_true
    bbox_gt = bbox_true[:]

    match = []

    for bbox in boxes_sorted:
        for gt in bbox_gt:
            #compute iou
            iou = calculate_iou(bbox[0], gt[0])
            if iou > threshold_iot and bbox[1] == gt[1]:
                #remove bbox from gt
                bbox_gt.remove(gt)

                #append bbox to match
                match.append(True)
                break
        else:
            match.append(False)
        
    #calculate precision and recall
    precision = [round(sum(match[:i+1]) / (i+1),2) for i in range(len(match))]
    recall = [round(sum(match[:i+1]) / len(bbox_true),2) for i in range(len(match))]

    #calculate average precision
    average_precision = 0
    for i in range(len(precision)-1):
        average_precision += (recall[i+1] - recall[i]) * precision[i+1]
    average_precision += recall[0] * precision[0]

    if not plot:
        return average_precision, precision, recall
    else:
        # Plot precision-recall curve
        figure, ax = plt.subplots()
        sns.scatterplot(x = recall, y = precision, marker = '*', s = 200, ax=ax)
        for i in range(len(precision)):
            ax.annotate(i+1, (recall[i], precision[i]))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        figure.savefig('/zhome/6d/e/184043/mario/DLCV/4/Results/mAP_curve.png')
        
        return round(average_precision,2), precision, recall


def calculate_iou(bbox1 : list[float], bbox2 : list[float]) -> float:
    # bbox : [x, y, w, h]    

    #Unpack bboxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    #Calculate areas of bboxes
    a1 = w1 * h1
    a2 = w2 * h2

    #Calculate intersection box corners, we denote intersection by z
    zx1, zy1 = max(x1, x2), max(y1, y2) #Bottom lefthand corner of intersection box
    zx2, zy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2) #Top righthand corner of intersection box
    zw, zh = zx2 - zx1, zy2 - zy1 #intersection width and height

    # Calculate intersection area and union area
    za = max(0, zw) * max(0, zh) #intersection area
    ua = a1 + a2 - za #union area, u denotes union

    # Calculate IOU
    iou = za / ua

    return float(iou)