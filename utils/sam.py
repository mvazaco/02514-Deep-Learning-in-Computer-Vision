from loss_functions import dice_coef, IoU, accuracy, specificity, sensivity

def plot_sam(target, model_output, sam_mask):
    return figure

def sam_metrics(sam_mask):
    sam_dict = {'dice': [], 'IoU': [], 'accuracy': [], 'sensivity', 'specificity': [], 'loss': []}
    return sam_dict

def sam_mask_processor(cut_out):
    cut_out_gray = rgb2gray(cut_out)
    sam_mask = (cut_out_gray != 1).astype(np.uint8)
    plt.imshow(sam_mask, cmap='gray')
    return sam_mask


checkpoint_path = 'path/to/your/checkpoint.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

