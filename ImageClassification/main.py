import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Normalize
import argparse
import json

from train import train
from models import BaselineCNN, BaselineCNN_w_dropout
from dataloader import Hotdog_NotHotdog


def main():

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model_options = {
        "BaselineCNN": BaselineCNN,
        "BaselineCNN_w_dropout": BaselineCNN_w_dropout
    }
    optimizer_options = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=model_options.keys(), default="BaselineCNN")
    parser.add_argument("--optimizer", type=str, choices=optimizer_options.keys(), default="Adam")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--augmentation", type=int, default=1)
    args = parser.parse_args()


    size = 128
    transform = [transforms.Resize((size, size))]
    if args.augmentation == 1:
        transform.append(transforms.RandomRotation(20))
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=(0.5226, 0.4412, 0.3585), std=(0.2253, 0.2294, 0.2339)))
                
    train_transform = transforms.Compose(transform)                              
    test_transform  = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5226, 0.4412, 0.3585),
                                                            std=(0.2253, 0.2294, 0.2339))
                                        ])

    batch_size = 64
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
    testset  = Hotdog_NotHotdog(train=False, transform=test_transform)
    
    # Init network
    model = model_options[args.model]()
    model.to(device)

    ### OPTIMIZER
    lr = args.lr
    L2 = 1e-3
    b1 = 0.7
    optimizer = optimizer_options[args.optimizer](model.parameters(), lr=lr)
    #optimizer = optimizer_options[args.optimizer](model.parameters(), lr=lr, weight_decay=L2)
    #optimizer = optimizer_options[args.optimizer](model.parameters(), lr=lr, betas=(b1, 0.999))

    epochs = args.epochs
    model_name = args.model
    opt_name = args.optimizer
    aug = args.augmentation
    out_dict = train(
        model=model, optimizer=optimizer, trainset=trainset, testset=testset, num_epochs=epochs, batch_size=batch_size, save_weights=False,
        model_name=model_name, optimizer_name=opt_name, lr_n=lr, aug=aug
    )
    file_path_template  = 'results/{}_{}_epochs={}_lr_{}_aug={}.json'
    file_path = file_path_template.format(model_name, opt_name, epochs, lr, aug)
    #file_path_template  = 'results/{}_{}_epochs={}_lr_{}_L2={}.json'
    #file_path = file_path_template.format(model_name, opt_name, epochs, lr, L2)
    #file_path_template  = 'results/{}_{}_epochs={}_lr_{}_L2={}_batchsize=128.json'
    #file_path = file_path_template.format(model_name, opt_name, epochs, lr, L2)
    #file_path_template  = 'results/{}_{}_epochs={}_lr_{}_aug={}_L2={}.json'
    #file_path = file_path_template.format(model_name, opt_name, epochs, lr, aug, L2)
    #file_path_template  = 'results/{}_{}_epochs={}_lr_{}_b1={}.json'
    #file_path = file_path_template.format(model_name, opt_name, epochs, lr, b1)
    with open(file_path, 'w') as json_file:
        json.dump(out_dict, json_file)

def analyze_data(trainset, batch_size):
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    mean = 0.
    std = 0.
    for images, _ in train_loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)  
    
    print(mean, std)

    
if __name__ == "__main__":
    main()
