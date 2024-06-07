#%%
import argparse
import sys

import torch
import yaml
from models import MaskGit as VQGANTransformer
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sys.argv = ['get_best.py']
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')#cuda
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for MaskGIT')
    
    parser.add_argument('--load-ckpt-linear-path', type=str, default='./data-linear/epoch=30.ckpt', help='load ckpt')
    parser.add_argument('--load-ckpt-square-path', type=str, default='./data-square/epoch=30.ckpt', help='load ckpt')
    parser.add_argument('--load-ckpt-cosine-path', type=str, default='./data-cosine/epoch=30.ckpt', help='load ckpt')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    
    model_linear = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
    ckpt_linear = torch.load(args.load_ckpt_linear_path)
    all_loss = torch.Tensor(ckpt_linear['all_loss'])
    min_linear = torch.min(all_loss, dim=0)
    indices_linear = min_linear.indices

    # Plot all loss
    plt.plot(all_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('linear')
    plt.show()

    model_square = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
    ckpt_square = torch.load(args.load_ckpt_square_path)
    all_loss = torch.Tensor(ckpt_square['all_loss'])
    min_square = torch.min(all_loss, dim=0)
    indices_square = min_square.indices

    # Plot all loss
    plt.plot(all_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('square')
    plt.show()


    model_cosine = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
    ckpt_cosine = torch.load(args.load_ckpt_cosine_path)
    all_loss = torch.Tensor(ckpt_cosine['all_loss'])
    min_cosine = torch.min(all_loss, dim=0)
    indices_cosine = min_square.indices

    # Plot all loss
    plt.plot(all_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('cosine')
    plt.show()





# %%
