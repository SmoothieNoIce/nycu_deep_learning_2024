import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.current_epoch = 0
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.all_loss = []
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)
        
    def train(self, train_loader, val_loader):
        for epoch in range(args.start_from_epoch+1, args.epochs+1):
            self.current_epoch = epoch
            self.train_one_epoch(train_loader)
            self.eval_one_epoch(val_loader)
            self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            self.optim.zero_grad()
            self.scheduler.step()

    def train_one_epoch(self, train_loader):
        for images in (pbar := tqdm(train_loader, ncols=120)):
            # batch_size = 10, channels = 3, width = 64, height = 64
            imgs = images.to(device=args.device)
            logits, target = self.model(imgs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.tqdm_bar('train', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        pass
    
    @torch.no_grad()
    def eval_one_epoch(self, val_loader):
        loss = 0
        total = 0
        for images in (pbar := tqdm(val_loader, ncols=120)):
            # batch_size = 10, channels = 3, width = 64, height = 64
            imgs = images.to(device=args.device)
            logits, target = self.model(imgs)
            loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            total +=1
            
        self.all_loss.append(loss/total)
        pass
    
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
    
    def save(self, path):
        torch.save({
            "state_dict": self.model.state_dict(),
            "lr"        : self.scheduler.get_last_lr()[0],
            "last_epoch": self.current_epoch,
            "all_loss": self.all_loss,
        }, path)
        print(f"save ckpt to {path}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.1)
        return optimizer,scheduler


if __name__ == '__main__':
    sys.argv = ['training_transformer.py', '--save_root', './data']
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()
    
    os.makedirs(args.save_root, exist_ok=True)

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    train_transformer.train(train_loader, val_loader)