import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
# Our imports
from Utility import generate_proposals_and_labels
from Utility import plot_images_jupyter
from Utility import WasteDatasetImages
from Models import ResNet50
from train import train


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_options = {
    "ResNet": ResNet50
}
optimizer_options = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam
}
loss_options = {
    "CrossEntropy": torch.nn.CrossEntropyLoss
}
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=model_options.keys(), default="ResNet")
parser.add_argument("--optimizer", type=str, choices=optimizer_options.keys(), default="Adam")
parser.add_argument("--loss", type=str, choices=loss_options.keys(), default="CrossEntropy")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()


#Load Data
data_path = '/dtu/datasets1/02514/data_wastedetection'
dataset = WasteDatasetImages(data_path, transform=transforms.ToTensor(), resize=(512,512))
num_classes = dataset.num_categories()

train_dataset_size = int(0.75 * len(dataset))
test_dataset_size  = len(dataset) - train_dataset_size
train_dataset, test_dataset = random_split(dataset, [train_dataset_size, test_dataset_size])
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset,  batch_size=1, shuffle=False)

max_proposals_per_image = 1000
# num_images_to_process_train = len(train_dataset)
# num_images_to_process_test = len(test_dataset)

## Quick limit for debugging/testing
num_images_to_process_train = 4 #Amount of train images to process

mode = 'quality'
patch_size = (224,224)

_, _, train_proposals_patches, train_patches_labels, _, _ = generate_proposals_and_labels(train_dataloader, mode, num_images_to_process_train, max_proposals_per_image, img_shape=patch_size)
# _, _, train_proposals_patches, train_patches_labels, _, _ = generate_proposals_and_labels(train_dataloader, mode, num_images_to_process_train, max_proposals_per_image, img_shape=patch_size)


training_size = int(0.75 * len(train_proposals_patches))
validation_size   = len(train_proposals_patches) - training_size 
train_patches_and_labels, val_patches_and_labels = random_split(list(zip(train_proposals_patches,train_patches_labels)), [training_size, validation_size])

batch_size = 256
train_patches_dataloader = DataLoader(train_patches_and_labels, batch_size=batch_size, shuffle=True)
val_patches_dataloader   = DataLoader(val_patches_and_labels,   batch_size=batch_size, shuffle=False)


# Model
model = model_options[args.model](num_classes=num_classes).to(device)
# Optimizer
optimizer = optimizer_options[args.optimizer](model.parameters(), lr=args.lr)
# Loss function
loss = loss_options[args.loss]().to(device)


# Training
out_dict, model = train(model, optimizer, 100, loss, train_patches_dataloader, val_patches_dataloader, len(train_patches_dataloader))

checkpoints_path = '/zhome/6d/e/184043/mario/DLCV/4/Training/Checkpoints/prueba_ratio2_SS{}_{}_lr={}_epochs={}_batchsize={}_images={}.pth'.format(mode, args.optimizer, args.lr, args.epochs, batch_size, num_images_to_process_train)
torch.save(model.state_dict(), checkpoints_path)

path = '/zhome/6d/e/184043/mario/DLCV/4/Training/out_dicts/prueba_ratio2_SS{}_{}_lr={}_epochs={}_batchsize={}_images={}.json'
path = path.format(mode, args.optimizer, args.lr, args.epochs, batch_size, num_images_to_process_train)
with open(path, 'w') as json_file:
    json.dump(out_dict, json_file)