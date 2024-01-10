import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.nn import functional as F
import torch
import torch.nn as nn
import glob

# Our imports
from Dataloader import WasteDatasetImages
from Utility import generate_proposals_and_labels
from Utility import plot_images_jupyter, plot_image_with_boxes
from Utility import no_max_supression, mean_average_precision
from Models import ResNet50


def calculate_iou(box1, box2):
    """Calculate intersection over union (IoU) between two bounding boxes with the format [x, y, width, height]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_w = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    intersect_h = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    intersection = intersect_w * intersect_h

    union = w1*h1 + w2*h2 - intersection
    return intersection / union

def calculate_ap(recall, precision):
    """Calculate the average precision (AP)."""
    # Append sentinel values at the end
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    # Non-maximum suppression
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Integrate the area under the precision-recall curve
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def calculate_map_per_class(real_bbox_label_image_vis, predicted_bbox_label_image_vis):
    """Calculate the mean average precision (mAP) for each class."""
    # Get all unique classes
    all_classes = set([label for item in real_bbox_label_image_vis for label in item['label']])

    # Initialize dictionary to hold AP for each class
    aps = {cls: [] for cls in all_classes}

    for real, pred in zip(real_bbox_label_image_vis, predicted_bbox_label_image_vis):
        # Sort predictions by confidence score
        pred = sorted([item for sublist in predicted_bbox_label_image_vis for item in sublist['p']], reverse=True)

        # Calculate AP for each class
        for cls in all_classes:
            # Filter boxes by class
            real_boxes_cls = [box for i, box in enumerate(real['bbox']) if real['label'][i] == cls]
            # Filter the bounding boxes by class
            pred_boxes_cls = [item for item in predicted_bbox_label_image_vis if item['label'] == cls]
            # Sort the filtered bounding boxes by 'p' value
            pred_boxes_cls = sorted(pred_boxes_cls, key=lambda x: x['p'], reverse=True)

            # Initialize variables
            num_real = len(real_boxes_cls)
            num_pred = len(pred_boxes_cls)
            tp = np.zeros(num_pred)
            fp = np.zeros(num_pred)
            ious = np.zeros((num_real, num_pred), dtype=np.float32)

            # Calculate IoU for each pair of predicted/real boxes
            for i in range(num_real):
                for j in range(num_pred):
                    ious[i, j] = calculate_iou(real_boxes_cls[i], pred_boxes_cls[j])

            # Determine true positives and false positives
            for j in range(num_pred):
                # Find the best matching real box
                best_i = np.argmax(ious[:, j])
                if ious[best_i, j] > 0.5:
                    tp[j] = 1
                    # Remove this real box and prediction
                    ious[best_i, :] = 0
                    ious[:, j] = 0
                else:
                    fp[j] = 1

            # Calculate precision and recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            recall = tp / num_real if num_real != 0 else tp
            precision = tp / (tp + fp)

            # Calculate average precision
            ap = calculate_ap(recall, precision)
            aps[cls].append(ap)

    # Calculate mean average precision (mAP) for each class
    map_per_class = {cls: np.mean(aps[cls]) for cls in all_classes}

    return map_per_class


if __name__ == "__main__":

    threshold_iou = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = '/dtu/datasets1/02514/data_wastedetection'
    dataset = WasteDatasetImages(data_path, transform=transforms.ToTensor(), resize=(512,512))
    num_classes = dataset.num_categories()

    model = ResNet50(num_classes)
    # model_path = '/zhome/6d/e/184043/mario/DLCV/4/Training/Checkpoints/SSquality_Adam_lr=0.001_epochs=100.pth'
    model_path =  '/zhome/6d/e/184043/mario/DLCV/4/Training/Checkpoints/ratio1_SSquality_Adam_lr=0.001_epochs=100.pth'
    model.load_state_dict(torch.load(model_path))
    # model.to(device)

    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_dataloader = DataLoader(test_dataset,  batch_size=1, shuffle=False)

    max_proposals_per_image = 1000 
    # num_images_to_process_test = len(test_dataset)
    num_images_to_process_test = 8


    mode = 'quality'
    patch_size = (128,128)
    _, proposals_box_list, resized_images, proposals_label,  _, image_idx = generate_proposals_and_labels(test_dataloader, mode, num_images_to_process_test, max_proposals_per_image, img_shape=patch_size)

    resized_images_array = np.array(resized_images, dtype=np.float32)
    resized_images_tensor = torch.from_numpy(resized_images_array)

    raw_scores = model(resized_images_tensor)

    # Convert raw scores to probabilities
    probabilities = F.softmax(raw_scores, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)
    predicted_labels = predicted_labels.cpu().numpy()

    # bbox_label_pairs = []
    # for i, (bbox, label) in enumerate(zip(proposals_box_list, predicted_labels.tolist())):
    #     if label != 0:
    #         probability = probabilities[i][label].item()
    #         bbox_label_pairs.append([bbox, label, probability])
    # probabilities[0][7].item()
    
    ctest = 0
    real_bbox_label_pairs = []      # list[list[float], int]
    real_bbox_label_image_vis = []  # for visualization purposes
    predicted_bbox_label_vis  = [] # for visualization purposes

    counting = 0
    visited  = 0

    for batch in test_dataloader:
        if ctest >= num_images_to_process_test:
            break
        images, bboxes, labels = batch
        for idx in range(len(bboxes)):
            bboxes[idx] = [coord.item() for coord in bboxes[idx]]  # Convert tensor to scalar
            real_bbox_label_pairs.append([bboxes[idx], labels[idx].item()])
            
        real_bbox_label_image_vis.append({
            "image": images[0].numpy(),  # Convert tensor to numpy array
            "bbox": bboxes,
            "label": [label.item() for label in labels]
        })
        bboxes, labels, predictions_p = [],[],[]
        for _ in range(image_idx.count(counting)):
            if proposals_label[visited] == 0:
                visited += 1
                continue
            bboxes.append(proposals_box_list[visited])
            labels.append(predicted_labels[visited])
            predictions_p.append(probabilities[visited].max().item())
            visited += 1
        predicted_bbox_label_vis.append({
            "image": images[0].numpy(),  # Convert tensor to numpy array
            "bbox": bboxes,
            "label": labels,
                "p": predictions_p
        })
        counting += 1
        ctest += 1
    
    file_count = len(glob.glob("/zhome/6d/e/184043/mario/DLCV/4/Results/*"))
    for idx in range(num_images_to_process_test):
        plot_image_with_boxes(real_bbox_label_image_vis[idx], predicted_bbox_label_vis[idx], f"cambio_GT_vs_pred_{idx+file_count}")


    # # Calculate mAP for each class
    # map_per_class = calculate_map_per_class(real_bbox_label_image_vis, predicted_bbox_label_vis)

    # # Print mAP for each class
    # for cls, map_score in map_per_class.items():
    #     print(f"mAP for class {cls}: {map_score}")