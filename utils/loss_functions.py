import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
    
def dice_coef(output, target):
    output = F.sigmoid(output) 
    predicted = (output > 0.5).to(torch.int)

    predicted = predicted.view(-1)
    target = target.view(-1)
    
    smooth = 1.
    intersection = (predicted * target).sum()                            
    dice = (2.*intersection + smooth)/(output.sum() + target.sum() + smooth)  
    return dice

def IoU(output, targets):
    output = F.sigmoid(output) 
    predicted = (output > 0.5).to(torch.int)
    
    predicted = predicted.view(-1)
    targets = targets.view(-1)
    
    intersection = (predicted * targets).sum()
    total = (predicted + targets).sum()
    union = total - intersection

    smooth=1
    IoU = (intersection + smooth) / (union + smooth)
    return IoU

def accuracy(output, targets):
    output = F.sigmoid(output) 
    predicted = (output > 0.5).to(torch.int)
    
    predicted = predicted.view(-1)
    targets = targets.view(-1)
    
    correct = (predicted == targets).sum()
    total = predicted.shape[0]
    return correct / total

def sensivity(output, targets):
    output = F.sigmoid(output) 
    predicted = (output > 0.5).to(torch.int)
    
    predicted = predicted.view(-1)
    targets = targets.view(-1)
    
    tp = ((predicted == 1) & (targets == 1)).sum()
    fn = ((predicted == 0) & (targets == 1)).sum()
    return tp / (tp + fn)

def specificity(output, targets):
    output = F.sigmoid(output) 
    predicted = (output > 0.5).to(torch.int)
    
    predicted = predicted.view(-1)
    targets = targets.view(-1)
    
    tn = ((predicted == 0) & (targets == 0)).sum()
    fp = ((predicted == 1) & (targets == 0)).sum()
    return tn / (tn + fp)