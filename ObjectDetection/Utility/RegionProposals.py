import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Our imports
from Dataloader import WasteDatasetImages


def generate_proposals_and_labels(dataloader, mode, num_images_to_process, max_proposals_per_image, img_shape = (224,224)):
    ########################################
    # Function to generate proposals and labels for train set

    # INPUTS:
    # - dataloader: dataloader for train set
    # - ss: selective search object
    # - num_images_to_process: number of images to process
    # - max_proposals_per_image: maximum number of proposals per image
    # - img_shape: shape of the proposals/candidates to resize to

    # OUTPUTS:
    # - data_list: list of tuples (bbox, label) for each image
    # - proposals_box_list: bounding boxes for all the images
    # - resized_images: list of images for all the images
    # - proposals_labels: list of labels for all the images
    # - images: tensor of original images
    # - image_idx: list of image index for each proposal
    ########################################
 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    data_list           = []
    images              = []
    resized_images      = []
    proposals_box_list  = []
    proposals_labels    = []
    image_idx           = []
    
    counter = 0
    for image, gt_bboxes, gt_label in tqdm(dataloader, desc=f"Processing images", total=num_images_to_process, leave=True):
        if counter >= num_images_to_process:
            break

        img = image.squeeze().permute(1, 2, 0).numpy()
        images.append(img)
        
        ss.setBaseImage(img)
        if mode == 'fast':
            ss.switchToSelectiveSearchFast() # Switch to fast but low-quality segmentation
        elif mode == 'quality':
            ss.switchToSelectiveSearchQuality()
        rects = ss.process() # Run selective search on the input image

        proposal_bbox   = []
        proposal_image  = []

        for i, rect in enumerate(rects): # loop for each region candidate in 1 image
            if i >= max_proposals_per_image: 
                break
            x, y, width, height = rect
            bbox = [int(x), int(y), int(width), int(height)]
            proposal_bbox.append(bbox)
            
            proposal_img = img[y:y+height, x:x+width]
            proposal_img = cv2.resize(proposal_img, img_shape)
            proposal_img = np.transpose(proposal_img, [2,0,1]) # Rearrange axis
            proposal_image.append(proposal_img)

        labels = assign_labels(proposal_bbox, gt_bboxes, gt_label, image)
        
        # Remove gray-zones (inbetween threshold)
        for i, label in reversed(list(enumerate(labels))):
            if label == -1:
                del labels[i]
                del proposal_bbox[i]
                del proposal_image[i]
        
        # Limit background patches, ratio 1 to 1 non background 
        num_non_bg = len(labels) - labels.count(0)
        num_bg     = labels.count(0)
        limit_bg   = num_non_bg * 2 #RATIO
        if limit_bg < num_bg: # Too many backgrounds
            transfer_labels = []
            transfer_bboxes = []
            transfer_images = []
            for idx in range(len(labels)):
                if labels[idx] == 0 and limit_bg == 0:
                    continue
                elif labels[idx] == 0:                  
                    limit_bg -= 1
                transfer_labels.append(        labels[idx])
                transfer_bboxes.append( proposal_bbox[idx])
                transfer_images.append(proposal_image[idx])
            labels          = transfer_labels
            proposal_bbox   = transfer_bboxes
            proposal_image  = transfer_images

        proposals_box_list.extend(proposal_bbox)
        resized_images.extend(proposal_image)
        proposals_labels.extend(labels)
        data_list.extend(list(zip(proposal_bbox, labels)))
        image_idx.extend([counter]*len(labels))
        assert len(proposal_bbox) != 0, proposal_bbox

        counter += 1
        
    images = torch.tensor(images)
    return data_list, proposals_box_list, resized_images, proposals_labels, images, image_idx

def assign_labels(proposals_bbox, gt_bboxes, gt_label, image, iou_threshold1=0.3, iou_threshold2=0.8):
    ########################################
    # Function to assign labels to proposals
    # OBS: In the code, this function only assigns labels for 1 input image.

    # INPUTS:
    # - proposals_bbox: list of bounding boxes for each proposal
    # - gt_bboxes: list of ground truth bounding boxes
    # - gt_label: list of ground truth labels
    # - image: original image

    # OUTPUTS:
    # - labels: list of labels for each proposal
    ########################################

    labels = []
    iou_scores = []
    
    max_iou_failsafe        = -1    # In case of no positive, use the best 
    max_iou_failsafe_info   = None  #/
    image = image.squeeze().permute(1, 2, 0).numpy()
    
    for proposal in proposals_bbox:
        proposal_x1, proposal_y1, proposal_w, proposal_h = proposal
        proposal_area = proposal_w * proposal_h

        max_iou         = -1 # Initialize max IOU score for each proposal
        max_label_idx   = -1

        for idx, bbox in enumerate(gt_bboxes):
            bbox_x1, bbox_y1, bbox_w, bbox_h = bbox

            # Compare boxes
            intersection_x1 = max(proposal_x1, bbox_x1)
            intersection_y1 = max(proposal_y1, bbox_y1)
            intersection_x2 = min(proposal_x1 + proposal_w, bbox_x1 + bbox_w)
            intersection_y2 = min(proposal_y1 + proposal_h, bbox_y1 + bbox_h)

            intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
            bbox_area = bbox_w * bbox_h
            union_area = proposal_area + bbox_area - intersection_area

            iou = intersection_area / union_area
            iou_scores.append(iou)

            # Get closest matching box 
            if iou > max_iou:
                max_iou         = iou
                max_label_idx   = idx
            #
            if iou > max_iou_failsafe:
                max_iou_failsafe        = iou
                max_iou_failsafe_info   = (proposal, idx)
            
        if   max_iou <= iou_threshold1: labels.append(0)                            # Object proposal is a negative example
        elif max_iou >= iou_threshold2: labels.append(int(gt_label[max_label_idx])) # Object proposal is a positive example
        else:                           labels.append(-1)                           # Object proposal is not considered

    if labels.count(0)+labels.count(-1) == len(labels):
        idx         = proposals_bbox.index(max_iou_failsafe_info[0])
        labels[idx] = int(gt_label[max_iou_failsafe_info[1]])
    # num_ones = len(labels)
    # print(f"Number of objects: {num_ones}")

    return labels

if __name__ == '__main__':

    from plots import plot_images_jupyter
 
    data_path = '/dtu/datasets1/02514/data_wastedetection'
    dataset = WasteDatasetImages(data_path, transform=transforms.ToTensor(), resize=(512,512))

    train_dataset_size = int(0.75 * len(dataset))
    test_dataset_size  = len(dataset) - train_dataset_size
    train_dataset, test_dataset = random_split(dataset, [train_dataset_size, test_dataset_size])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    max_proposals_per_image = 1000
    num_images_to_process_train = 8 #Amount of train images to plot object proposals

    patch_size = (224,224)
    train_data, train_proposals, train_proposals_image, train_label, images_og, image_idx = generate_proposals_and_labels(train_dataloader, num_images_to_process_train, max_proposals_per_image, img_shape=patch_size)
    plot_images_jupyter(images_og, image_idx, train_proposals, train_label)


