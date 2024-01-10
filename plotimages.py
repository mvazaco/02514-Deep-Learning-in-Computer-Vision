import matplotlib.pyplot as plt
import torch
import numpy as np

def plotimages(dataloader):
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(20,10))
    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(images[i].numpy()[0], 'gray')
        plt.title(labels[i].item())
        plt.axis('off')
    plt.show()
    path = 'figs/svhnImg.png'
    plt.savefig(path)
    
    return path

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def plotwrongimages(test_loader, model, model_name, optimizer, epochs, augmentation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    plt.figure(figsize=(20,10))
    f_neg_images = []
    f_pos_images = []
    f_pos = 0
    t_pos = 0
    t_neg = 0
    f_neg = 0
    for images, labels in test_loader:
        for image, label in zip(images, labels):

            unnorm = UnNormalize((0.4381, 0.4442, 0.4732), (0.1170, 0.1200, 0.1025))
            
            image = image.view(1, *image.shape)
            pred = model(image.to(device)).detach().cpu().numpy().item()
            label.item()
            
            bin_pred = (pred > 0.5)

            if bin_pred == 1:
                if label.item() == 0:
                    f_neg += 1
                    if len(f_neg_images) < 24:
                        f_neg_images.append((unnorm(image).permute(0,2,3,1).detach().cpu().numpy()[0], label, pred))
                else:
                    t_neg += 1
            else:
                if label.item() == 1:
                    f_pos += 1
                    if len(f_pos_images) < 24:
                        f_pos_images.append((unnorm(image).permute(0,2,3,1).detach().cpu().numpy()[0], label, pred))
                else:
                    t_pos += 1
                    
   # Plotting false postive
    count = 0
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 10))
    for i in range(4):
        for j in range(5):
            image, label, pred = f_pos_images[count]
            axes[i,j].imshow(image)
            axes[i,j].set_title(f"Predicted: {pred:.2f}, True: {label}", fontsize=10)
            axes[i,j].set_axis_off()
            count += 1
    plt.suptitle('False positive images', fontsize=15)
    plt.tight_layout()
    path_pos_template  = 'figs/f_pos_pred_{}_{}_epochs={}__aug={}.png'
    path_pos = path_pos_template.format(model_name, optimizer, epochs, augmentation)
    plt.savefig(path_pos)
    #plt.show()
    
     # Plotting false negative
    count = 0
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 10))
    for i in range(4):
        for j in range(5):
            image, label, pred = f_neg_images[count]
            axes[i,j].imshow(image)
            axes[i,j].set_title(f"Predicted: {pred:.2f}, True: {label}", fontsize=10)
            axes[i,j].set_axis_off()
            count += 1
    plt.suptitle('False negative images', fontsize=15)
    plt.tight_layout()
    path_neg_template  = 'figs/f_neg_pred_{}_{}_epochs={}_aug={}.png'
    path_neg = path_neg_template.format(model_name, optimizer, epochs, augmentation)
    plt.savefig(path_neg)
    #plt.show()

    return path_pos, path_neg, f_pos, t_pos, t_neg, f_neg


