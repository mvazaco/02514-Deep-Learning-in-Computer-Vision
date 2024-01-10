import os
import sys
import glob
import json
from time import time
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
# from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils import dice_coef, IoU, accuracy, specificity, sensivity
from utils import plot_results


def train(model, optimizer, loss_fn, batch_size,
          dataset, train_loader, test_loader,
          epochs=10, patience=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # scheduler = StepLR(optimizer, step_size=25, gamma=0.1) 

    out_dict = {'dice_train': [], 'dice_test': [],
                'IoU_train': [], 'IoU_test': [],
                'accuracy_train': [], 'accuracy_test': [],
                'sensivity_train': [], 'sensivity_test': [],
                'specificity_train': [], 'specificity_test': [],
                'loss_train': [], 'loss_test': []
    }

    num_epochs = epochs
    atl_test_loss = sys.maxsize
    countdown = patience

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_loss = []
        train_dice, train_IoU, train_acc, train_sen, train_spec = [], [], [], [], []
        for minibatch_no, (data, target, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fn(output, target)
            # print(loss.size())
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
            # scheduler.step()

            # Compute metrics for validating segmentation performance
            train_loss.append(loss.item())
            train_dice.append(dice_coef(output.to(torch.float), target.to(torch.float)).item())
            train_IoU.append(IoU(output.to(torch.float), target.to(torch.float)).item())
            train_acc.append(accuracy(output.to(torch.float), target.to(torch.float)).item())
            train_sen.append(sensivity(output.to(torch.float), target.to(torch.float)).item())
            train_spec.append(specificity(output.to(torch.float), target.to(torch.float)).item())
                
        #Compute metrics for test set
        model.eval()
        test_loss = []
        test_dice, test_IoU, test_acc, test_sen, test_spec = [], [], [], [], []
        for data, target, images_names in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)

            test_loss.append(loss_fn(output, target).cpu().item())
            test_dice.append(dice_coef(output.to(torch.float), target.to(torch.float)).item())
            test_IoU.append(IoU(output.to(torch.float), target.to(torch.float)).item())
            test_acc.append(accuracy(output.to(torch.float), target.to(torch.float)).item())
            test_sen.append(sensivity(output.to(torch.float), target.to(torch.float)).item())
            test_spec.append(specificity(output.to(torch.float), target.to(torch.float)).item())

        # print(train_dice)
        out_dict['loss_train'].append(np.mean(train_loss))
        out_dict['loss_test'].append(np.mean(test_loss))
        out_dict['dice_train'].append(np.mean(train_dice))
        out_dict['dice_test'].append(np.mean(test_dice))
        out_dict['IoU_train'].append(np.mean(train_IoU))
        out_dict['IoU_test'].append(np.mean(test_IoU))
        out_dict['accuracy_train'].append(np.mean(train_acc))
        out_dict['accuracy_test'].append(np.mean(test_acc))
        out_dict['sensivity_train'].append(np.mean(train_sen))
        out_dict['sensivity_test'].append(np.mean(test_sen))
        out_dict['specificity_train'].append(np.mean(train_spec))
        out_dict['specificity_test'].append(np.mean(test_spec))

        current_test_loss = out_dict['loss_test'][-1]
        
        if atl_test_loss > current_test_loss:
            atl_test_loss = current_test_loss
            countdown = patience
        else:
            countdown -= 1
            if countdown == 0:
                num_epochs = epoch
                print(f'Stopping at epoch {epoch} because test loss haven\'t improved in the last {patience} epochs')
                break
        
    figure = plot_results(data, output, target, images_names, batch_size, dataset)

    if dataset =='DRIVE':
        torch.save(model.state_dict(), '/zhome/6d/e/184043/DLCV/2/project/models/UNet_Adam_FocalLoss_0.001_aug_0.pth')

        images_names_path = '/zhome/6d/e/184043/DLCV/2/project/results/DRIVE_images_names_list'
        with open(images_names_path, 'w') as json_file:
            json.dump(images_names, json_file)
    else:
        torch.save(model.state_dict(), '/zhome/6d/e/184043/DLCV/2/project/models/PH2_UNet_Adam_FocalLoss_0.0001_aug_0.pth')
        
        images_names_path = '/zhome/6d/e/184043/DLCV/2/project/results/PH2_images_names_list'
        with open(images_names_path, 'w') as json_file:
            json.dump(images_names, json_file)

    return out_dict, figure