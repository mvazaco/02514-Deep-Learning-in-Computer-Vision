import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split


class PH2_dataset(torch.utils.data.Dataset):
    def __init__(self, transform, data_path='/dtu/datasets1/02514/PH2_Dataset_images'):
        self.transform = transform
        self.path = data_path
        self.names = os.listdir(self.path)
        self.items = sorted(glob.glob(data_path + '/*'))

    def __len__(self):
        return len(self.items)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
    
    def __getitem__(self, index):
        item_path = self.items[index]
        item_name = self.names[index]
        image = Image.open(os.path.join(item_path, f'{item_name}_Dermoscopic_Image', f'{item_name}.bmp')).convert("RGB")
        mask = Image.open(os.path.join(item_path, f'{item_name}_lesion', f'{item_name}_lesion.bmp'))

        seed = random.randint(0, 2**32)
        self._set_seed(seed)
        X = self.transform(image)
        self._set_seed(seed)
        Y = self.transform(mask)

        return X, Y, item_name
   

class DRIVE_dataset(torch.utils.data.Dataset):
    def __init__(self, transform, data_path='/dtu/datasets1/02514/DRIVE/training'):
        self.transform = transform
        self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
        self.mask_paths  = sorted(glob.glob(data_path + '/1st_manual/*.gif'))

    def __len__(self):
        return len(self.image_paths)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
        mask_path  = self.mask_paths[idx]
        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path)

        seed = random.randint(0, 2**32)
        self._set_seed(seed)
        X = self.transform(image)
        self._set_seed(seed)
        Y  = self.transform(mask)

        return X, Y, image_name


def get_data(dataset, train_percentage,  batch_size):
    data = dataset
    train_size = int(train_percentage * len(data))
    test_size  = len(data) - train_size 

    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size, shuffle=False)

    return data, train_loader, test_loader
