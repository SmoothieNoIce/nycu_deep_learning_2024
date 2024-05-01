# %%
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from oxford_pet import SimpleOxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import UResNet34

from utils import dice_coeff, plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    #print(img.shape)
    #print(img.shape[2])

    with torch.no_grad():
        
        output = net(img).cpu()
        output = F.interpolate(output, (img.shape[2], img.shape[3]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    sys.argv = ['predict.py' , '--model-select', '0', '-i' ,'./dataset/images/Abyssinian_1.jpg', '-o', 'output-1.jpg', '-m' ,'../saved_models/unet/DL_Lab3_UNet_312551013_謝竣宇.pth']
    #sys.argv = ['predict.py', '--model-select', '1', '-i' ,'./dataset/images/Abyssinian_1.jpg', '-o', 'output-1.jpg', '-m' ,'../saved_models/res/DL_Lab3_ ResNet34_UNet _312551013_謝竣宇.pth']
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', default=True, action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--model-select', '-d', type=int, default=0, help='Type of model')

    args, unknown = parser.parse_known_args()
    return args


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    path = os.getcwd()
    test_dataset = SimpleOxfordPetDataset(path + "/../dataset", "test")
    out_files = get_output_filenames(args)

    if args.model_select == 0:
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    else:
        net = UResNet34(n_channels=3, n_classes=args.classes)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict['state_dict'])

    logging.info('Model loaded!')
    dice_score = 0
    size = 0

    for i, data in enumerate(test_dataset):
        size += 1
        #logging.info(f'Predicting image  ...')
        img = data['image']
        mask_true = data['mask'].squeeze(0)
        
        mask_pred = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        transform1 = transforms.Compose([
            transforms.ToTensor()
            ]
        )
        mask_true_tensor = transform1(mask_true)
        mask_pred_tensor = transform1(mask_pred)
        dice_score += dice_coeff(mask_pred_tensor, mask_true_tensor, reduce_batch_first=False)

        if not args.no_save:
            pass
            #out_filename = out_files[i]
            #result = mask_to_image(mask, mask_values)
            #result.save(out_filename)
            #logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            #logging.info(f'Visualizing results for image , close to continue...')
            #plot_img_and_mask(img, mask_pred)
            pass
            
    dice_score = dice_score / size
    print(dice_score)
# %%
