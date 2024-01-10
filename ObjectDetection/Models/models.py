import torch
import torch.nn as nn
from torchvision import models

def ResNet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model