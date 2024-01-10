import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Own
from Dataloader import SUPERCATEGORIES, ALL_SUPERCATEGORIES
from Utility import no_max_supression, mean_average_precision


def plot_images_jupyter(images_og, image_idx, train_proposals, train_label, num_images = 8):
    # def_colors = ["red", "blue", "orange", "purple", "cyan", "pink", "olive", "yellow", "navy"]
    def_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'olive', 'navy', 'coral', 'gold', 'silver']
    num_images   = num_images
    counter      = 0
    prev_img_idx = image_idx[0]
    for idx in range(num_images):
        fig, ax = plt.subplots()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(images_og[idx])
        has_label = set()
        while True:
            if counter+1 == len(train_label) or image_idx[counter] != prev_img_idx:
                prev_img_idx = image_idx[counter]
                break
            lab = train_label[counter]
            x, y, width, height = train_proposals[counter]
            rect = patches.Rectangle((x, y), width, height, linewidth=min(3,2*lab+1), edgecolor=def_colors[lab], 
                                     facecolor='none', label = ALL_SUPERCATEGORIES[lab] if lab not in has_label else "")
            ax.add_patch(rect)
            has_label.add(lab)
            counter += 1
        ax.legend()
        fig.savefig(f"/zhome/6d/e/184043/mario/DLCV/4/object_proposals/Patches_{idx}.png")


def plot_image_with_boxes(gt_dict_info, pred_dict_info, name):

    threshold_iou = 0.3

    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    fig, ax = plt.subplots(1,2,)
    
    # for idx, (used_dict,text) in enumerate(zip([gt_dict_info, pred_dict_info],[,])):
    ax[0].imshow(gt_dict_info["image"].transpose(1, 2, 0))
    gt_info = []
    for i, (bbox, label) in enumerate(zip(gt_dict_info["bbox"], gt_dict_info["label"])):
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=colors[i % len(colors)], facecolor='none')
        ax[0].add_patch(rect)
        ax[0].text(x, y+h+15, str(label), color="white", fontsize=10, va="bottom", bbox=dict(facecolor='black', alpha=1))
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        ax[0].set_title("Ground Truth\n")
        gt_info.append([bbox, label])

    pred_bbox, pred_lab, pred_p = pred_dict_info["bbox"], pred_dict_info["label"], pred_dict_info["p"]
    pred_bboxes = [[bbox, lab, prop] for bbox, lab, prop in zip(pred_bbox, pred_lab, pred_p)]
    # raise Exception([print(b) for b in pred_bboxes])
    bbox_new = no_max_supression(pred_bboxes, 0.4)
    # raise Exception([print(b) for b in bbox_new])
    
    ax[1].imshow(pred_dict_info["image"].transpose(1, 2, 0))
    for i, (bbox, label, _) in enumerate(bbox_new):
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=colors[label % len(colors)], facecolor='none')
        ax[1].add_patch(rect)
        ax[1].text(x, y+h+15, str(label), color="white", fontsize=10, va="bottom", bbox=dict(facecolor='black', alpha=1))
        ax[1].xaxis.set_visible(False)
        ax[1].yaxis.set_visible(False)
        ax[1].set_title("Prediction\n")
    # raise Exception(len(gt_info), len(pred_bboxes),len(bbox_new))
    # raise Exception(gt_info,bbox_new)
    average_precision, precision, recall = mean_average_precision(gt_info, bbox_new,threshold_iot=threshold_iou, plot = False)

    plt.tight_layout()
    # fig.suptitle(f"Threshold IoU: {threshold_iou}\n Precision: {precision}\nAverage_precision: {average_precision}\nRecall: {recall}\n\n")
    fig.suptitle(f'Precision: {precision[-1]}\n Recall: {recall[-1]}')
    fig.savefig(f'/zhome/6d/e/184043/mario/DLCV/4/Results/{name}.png')