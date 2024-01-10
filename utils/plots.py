import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

def plot_samples(loader, dataset):
    data, target, _ = next(iter(loader))
    num_samples = 5
    figure = plt.figure()
    for k in range(num_samples):
        plt.subplot(2, num_samples, k+1)
        plt.imshow(np.transpose(data[k].numpy(), (1, 2, 0)))
        plt.title('Image')
        plt.axis('off')

        plt.subplot(2, num_samples, k+num_samples+1)
        plt.imshow(np.transpose(target[k].numpy(), (1, 2, 0)), cmap='gray')
        plt.title('Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return figure

def plot_results(images, output, target, images_names, batch_size, dataset):

    if dataset == 'PH2':
        output = F.sigmoid(output)
        predicted = (output > 0.5).to(torch.int)

        # plt.figure(figsize=(15, 5))
        figure = plt.figure()
        for k in range(3):

            plt.subplot(3, 3, k+1)
            image = images[k].cpu().numpy()
            plt.imshow(np.transpose(image, (1, 2, 0)))
            plt.title(f'{images_names[k]}')
            plt.axis('off')

            plt.subplot(3, 3, k+3+1)
            plt.imshow(predicted[k, 0].cpu().numpy(), cmap='gray')
            plt.title('Predicted')
            plt.axis('off')

            plt.subplot(3, 3, k+6+1)
            plt.imshow(target[k, 0].cpu().numpy(), cmap='gray')
            plt.title('Mask')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        
    else:
        # print(output.size())
        # print(target.size())
        # print(batch_size)
        batch_size=int(batch_size/2)

        # plt.imshow(output[0, 0].cpu().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()
        
        output = F.sigmoid(output)
        # plt.imshow(output[0, 0].cpu().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()
        
        predicted = (output > 0.5).to(torch.int)
        # plt.imshow(predicted[0, 0].cpu().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()
        
        # plt.imshow(target[0, 0].cpu().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()

        # plt.figure(figsize=(15, 5))
        figure = plt.figure()
        for k in range(batch_size):
            plt.subplot(2, batch_size, k+1)
            # plt.imshow(np.rollaxis(predicted[k].cpu().numpy, 0, 3), cmap='gray')
            plt.imshow(predicted[k, 0].cpu().numpy(), cmap='gray')
            plt.title('Predicted')
            plt.axis('off')

            plt.subplot(2, batch_size, k+batch_size+1)
            plt.imshow(target[k, 0].cpu().numpy(), cmap='gray')
            plt.title(f'{images_names[k]}')
            plt.axis('off')
        # plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))

        plt.tight_layout()
        # directory_path = '/zhome/6d/e/184043/DLCV/2/project/figs'
        # file_path = os.path.join(directory_path, 'result.png')
        # plt.savefig(file_path)
        plt.show()

    return figure

def plot_metrics(data):
    num_epochs = range(1, len(data['loss_train'])+1)
    
    figure = plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    plt.plot(num_epochs, data['loss_train'],      color='blue',   label='train')
    plt.plot(num_epochs, data['loss_test'], '--', color='blue',   label='test')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    # plt.xticks(np.arange(1, len(data['loss_train'])+1, 10), fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    plt.legend(fontsize=15)
    plt.grid()
    
    plt.subplot(3, 3, 2)
    plt.plot(num_epochs, data['dice_train'],      color='blue',   label='train')
    plt.plot(num_epochs, data['dice_test'], '--', color='blue',   label='test')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Dice coefficient', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    plt.legend(fontsize=15)
    plt.grid()
    
    plt.subplot(3, 3, 3)
    plt.plot(num_epochs, data['IoU_train'],      color='red',    label='train')
    plt.plot(num_epochs, data['IoU_test'], '--', color='red',    label='test')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('IoU', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    plt.legend(fontsize=15)
    plt.grid()
    
    plt.subplot(3, 3, 4)
    plt.plot(num_epochs, data['accuracy_train'],      color='orange',  label='train')
    plt.plot(num_epochs, data['accuracy_test'], '--', color='orange',  label='test')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0.5, 1])
    plt.legend(fontsize=15)
    plt.grid()
    
    plt.subplot(3, 3, 5)
    plt.plot(num_epochs, data['sensivity_train'],      color='green', label='train')
    plt.plot(num_epochs, data['sensivity_test'], '--', color='green', label='test')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Sensivity', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    plt.legend(fontsize=15)
    plt.grid()
    
    plt.subplot(3, 3, 6)
    plt.plot(num_epochs, data['specificity_train'],      color='orange',  label='train')
    plt.plot(num_epochs, data['specificity_test'], '--', color='orange',  label='test')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Specificity', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim([0.4, 1])
    plt.legend(fontsize=15)
    plt.grid()
    
    plt.tight_layout()
    plt.show()

    return figure
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        