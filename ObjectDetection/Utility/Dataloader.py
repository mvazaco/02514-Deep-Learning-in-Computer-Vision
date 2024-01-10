import torch
import os
import numpy as np
import json
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

ALL_SUPERCATEGORIES = ["Background", "Aluminium foil", "Bottle", "Bottle cap", "Broken glass", "Can", "Carton", "Cup", "Lid", "Other plastic", "Paper", "Plastic bag & wrapper", "Plastic container", "Pop tab", "Straw", "Styrofoam piece", "Unlabeled litter", "Cigarette"]
SUPERCATEGORIES = ["Background", "Bottle", "Bottle cap", "Can", "Carton", "Plastic bag & wrapper", "Unlabeled litter", "Cigarette"]
UNLABELED = SUPERCATEGORIES.index("Unlabeled litter")

class WasteDatasetImages(Dataset):
    def __init__(self, data_path, transform=None, resize=(224,224)):
        self.data_path = data_path
        with open(os.path.join(data_path, 'annotations.json')) as f:
            data = json.load(f)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor()
        ])
        self.img_info = data['images']
        self.annotation = data['annotations']
        self.categories = data['categories']
        self.resize = resize  # Specify the desired resize dimensions
    
    def __len__(self):
        return len(self.img_info)
    
    def __getitem__(self, idx):
        img_info = self.img_info[idx]
        img_id = img_info['id']
        img_file_name = img_info['file_name']
        img = Image.open(os.path.join(self.data_path, img_file_name))
        
        # Resize the image
        resized_img = img.resize(self.resize)
        transformed_img = self.transform(resized_img)
        
        patches = [ann for ann in self.annotation if ann['image_id'] == img_id]
        #print(f"Found {len(bboxes)} bounding boxes for image {img_id}")
        resized_bboxes = []
        labels = []
        for patch in patches:
            bbox = patch['bbox']
            resized_bbox = [
                bbox[0] * self.resize[0] / img.width,  # x
                bbox[1] * self.resize[1] / img.height,  # y
                bbox[2] * self.resize[0] / img.width,  # width
                bbox[3] * self.resize[1] / img.height  # height
            ]
            resized_bboxes.append(resized_bbox)

            supercat = self.categories[patch['category_id']]['supercategory']
            #if we're not using the class (not in supercategories list), set to unlabeled
            label = ALL_SUPERCATEGORIES.index(supercat) if supercat in ALL_SUPERCATEGORIES else UNLABELED
            labels.append(label)

        return transformed_img, resized_bboxes, labels
    
    def num_categories(self):
        return len(ALL_SUPERCATEGORIES)
    
    def category_name(self, label):
        return ALL_SUPERCATEGORIES[label]

