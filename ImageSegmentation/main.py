import sys
import os
import argparse
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms

#Our imports
from dataloaders import PH2_dataset, DRIVE_dataset, get_data
from models import EncDec, UNet
from utils import FocalLoss, plot_metrics, plot_samples
from train import train
from SAM import sam_plot


def main():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_options = {
        "PH2": PH2_dataset,
        "DRIVE": DRIVE_dataset
    }
    model_options = {
        "Baseline": EncDec,
        "UNet": UNet
    }
    optimizer_options = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "RMSprop": torch.optim.RMSprop
    }
    loss_options = {
        "BCE": nn.BCEWithLogitsLoss,
        "FocalLoss": FocalLoss
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=dataset_options.keys(), default="PH2")
    parser.add_argument("--model", type=str, choices=model_options.keys(), default="UNet")
    parser.add_argument("--optimizer", type=str, choices=optimizer_options.keys(), default="Adam")
    parser.add_argument("--loss", type=str, choices=loss_options.keys(), default="FocalLoss")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--augmentation", type=int, default=0)
    args = parser.parse_args()
    
    #Load Data
    if args.dataset == 'PH2':
        batch_size = 32
        size = 256
        transform = [transforms.Resize((size, size))]
        if args.augmentation == 1:
            transform.append(transforms.RandomRotation(20))
            transform.append(transforms.RandomHorizontalFlip())
            transform.append(transforms.RandomVerticalFlip()),
        transform.append(transforms.ToTensor())
        transform = transforms.Compose(transform)

        dataset = PH2_dataset(transform)
        _, train_loader, test_loader = get_data(dataset, train_percentage=0.75, batch_size=batch_size)
    
    if args.dataset == 'DRIVE':
        batch_size = 6
        size = 512
        transform = [transforms.Resize((size, size))]
        if args.augmentation == 1:
            transform.append(transforms.RandomRotation(20))
            transform.append(transforms.RandomHorizontalFlip())
            transform.append(transforms.RandomVerticalFlip()),
        transform.append(transforms.ToTensor())
        transform = transforms.Compose(transform)

        dataset = DRIVE_dataset(transform)
        _, train_loader, test_loader = get_data(dataset, train_percentage=0.75, batch_size=batch_size)
        
    # Plot samples  
    figure = plot_samples(train_loader, args.dataset)
    directory_path = '/zhome/6d/e/184043/DLCV/2/project/figs'
    file_path = os.path.join(directory_path, 'sample_{}_aug_{}.png'.format(args.dataset, args.augmentation))
    figure.savefig(file_path)

    # Init network
    model_str = args.model
    model = model_options[model_str](n_channels=3, n_classes=1, image_size=size).to(device)
    
    # Optimizer
    optimizer_str = args.optimizer
    lr = args.lr
    optimizer = optimizer_options[optimizer_str](model.parameters(), lr=lr)
    
    # Train
    epochs = args.epochs
    loss_fn  = loss_options[args.loss]()
    loss_str = args.loss
    # aug = args.augmentation
    out_dict, figure = train(model=model, optimizer=optimizer, loss_fn=loss_fn, batch_size=batch_size,
                     dataset=args.dataset, train_loader=train_loader, test_loader=test_loader,
                     epochs=epochs)

    directory_path = '/zhome/6d/e/184043/DLCV/2/project/figs'
    file_path = os.path.join(directory_path, 'outputs/{}_{}_{}_{}_epochs={}_lr_{}_aug_{}.png'.format(args.dataset, model_str, optimizer_str, loss_str, epochs, lr, args.augmentation))
    figure.savefig(file_path)
    
    figure = plot_metrics(out_dict)
    directory_path = '/zhome/6d/e/184043/DLCV/2/project/figs'
    file_path = os.path.join(directory_path, 'metrics/{}_{}_{}_{}_epochs={}_lr_{}_aug_{}.png'.format(args.dataset, model_str, optimizer_str, loss_str, epochs, lr, args.augmentation))
    figure.savefig(file_path)
    
    file_path_template  = 'results/{}_{}_{}_{}_epochs={}_lr_{}_aug_{}.json'
    file_path = file_path_template.format(args.dataset, model_str, optimizer_str, loss_str, epochs, lr, args.augmentation)
    with open(file_path, 'w') as json_file:
        json.dump(out_dict, json_file)
        
if __name__ == main():
    main()