# %%
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from models.unet import UNet
from models.resnet34_unet import UResNet34
from matplotlib import pyplot as plt

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    
    return NotImplemented

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def plot_img_and_mask(img, mask):
    img = img.transpose((1, 2, 0))
    print(img.shape)
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_model = UNet(n_channels=3, n_classes=2, bilinear=False)
    unet_model = unet_model.to(memory_format=torch.channels_last)
    unet_state_dict = torch.load('../saved_models/unet/checkpoint_epoch10.pth', map_location=device)
    unet_model.load_state_dict(unet_state_dict['state_dict'])
    unet_epoch_loss_all = unet_state_dict["epoch_loss_all"]
    unet_val_loss_all = unet_state_dict["val_loss_all"]
    res_model = UResNet34(n_channels=3, n_classes=2)
    res_model = res_model.to(memory_format=torch.channels_last)
    res_state_dict = torch.load('../saved_models/res/checkpoint_epoch10.pth', map_location=device)
    res_model.load_state_dict(res_state_dict['state_dict'])
    res_epoch_loss_all = res_state_dict["epoch_loss_all"]
    res_val_loss_all = res_state_dict["val_loss_all"]
    # Plotting loss values
    plt.plot(unet_epoch_loss_all, label='UNet Epoch Loss', color='blue')
    plt.plot(unet_val_loss_all, label='UNet Validation Dice Score', color='cyan')
    plt.plot(res_epoch_loss_all, label='UResNet50 Epoch Loss', color='red')
    plt.plot(res_val_loss_all, label='UResNet50 Validation Dice Score', color='magenta')

    # Adding labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Dice Comparison')
    plt.legend()

    # Displaying the plot
    plt.show()
# %%
