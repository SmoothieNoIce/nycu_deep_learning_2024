from functools import partial
import os
import random
import sys
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from matplotlib import transforms
from torchvision import utils as vutils
from torch.utils.data import DataLoader, Subset
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from diffusers import DDPMScheduler
from torch.utils.tensorboard import SummaryWriter

from dataset import iclevrDataset
from evaluator import evaluation_model
from ddpm import ConditionedUNet
import wandb

class TrainDDPM:
    def __init__(self, args): 
        self.args = args
        self.current_epoch = 0
        self.best_acc = 0

        self.unet = ConditionedUNet(args).to(args.device)
        self.noise_scheduler = DDPMScheduler(args.timesteps)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.unet.parameters(), lr = args.lr)

        self.writer = SummaryWriter(args.log)

    def load_state_dict(self, state_dict):
        self.unet.load_state_dict(state_dict['state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.current_epoch = state_dict['last_epoch']
        self.best_acc = state_dict['best_acc']

    def train(self, train_loader, val_loader):
        for epoch in range(self.args.start_from_epoch + 1, self.args.epochs + 1):
            self.current_epoch = epoch
            loss = self.train_one_epoch(epoch, train_loader)

            self.writer.add_scalar('Train Epoch Loss', loss, epoch)
            print(f'Train Epoch_{epoch} loss: {loss}\n')

            avg_acc = self.eval_one_epoch(epoch, val_loader)
            print(f'Testing acc: {avg_acc}\n')
            eval_metric = {"Training Epoch Loss": loss,"Testing acc": avg_acc}
            wandb.log(eval_metric)

            is_best = False
            if avg_acc > self.best_acc:
                self.best_acc = avg_acc
                is_best = True
                print(f'New Best Acc! Epoch_{epoch} acc: {avg_acc}\n')

            self.save(loss, avg_acc, is_best)
    
    def train_one_epoch(self, epoch, train_loader):
        iters = 0
        self.unet.train()
        total_loss = 0
        for i, (image, cond) in enumerate(pbar := tqdm(train_loader)):
            # batch_size = 64, channels = 3, width = 64, height = 64

            self.optimizer.zero_grad()

            device = args.device
            real_image = image.to(device)
            cond = cond.to(device)
            batch_size = image.size(0)

            noise = torch.randn_like(real_image)
            timesteps = torch.randint(0, args.timesteps - 1, (batch_size,)).long().to(device)
            noisy_img = self.noise_scheduler.add_noise(real_image, noise, timesteps)

            pred = self.unet(noisy_img, timesteps, cond)

            loss = self.criterion(pred, noise)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if iters % 100 == 0:
                self.writer.add_scalar('Train Step loss', loss.item(), iters)
                wandb.log({"Training Step loss": loss.item()})

            pbar.set_description('[%d/%d][%d/%d] Loss: %.4f'
                % (epoch, args.epochs, i, len(train_dataloader),
                    loss.item()))
            pbar.refresh()

            iters += 1
        return total_loss / len(train_loader)

    @torch.no_grad()
    def eval_one_epoch(self, epoch, test_dataloader):
        args = self.args
        evaluator = evaluation_model()
        self.unet.eval()
        with torch.no_grad():
            for i, cond in enumerate(test_dataloader):
                cond = cond.to(args.device)
                batch_size = cond.size(0)
                noise = torch.randn(batch_size, 3, 64, 64, device=args.device)

                for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                    # Get model predict
                    residual = self.unet(noise, t, cond)
                    
                    # Update sample with step
                    noise = self.noise_scheduler.step(residual, t, noise).prev_sample

                acc = evaluator.eval(noise, cond)
                vutils.save_image(vutils.make_grid(noise, nrow = 8, normalize = True),
                    '%s/epoch_%d_fake_test.png' % ( args.outf, epoch),
                    normalize=True)
            print(f'Sample : {acc*100:.2f}%')
        return acc

    def save(self, loss, avg_acc, is_best=False):
        if is_best:
            text = f"{self.args.outf_checkpoint}/unet_epoch_{self.current_epoch}_{avg_acc}_best.pth",
        else:
            text = f"{self.args.outf_checkpoint}/unet_epoch_{self.current_epoch}_{avg_acc}.pth",
        torch.save(
            {
                "state_dict": self.unet.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr": self.args.lr,
                "last_epoch": self.current_epoch,
                "avg_acc": avg_acc,
                "loss": loss,
                "best_acc": self.best_acc,
            },
            text,
        )
        print(f"save ckpt")

if __name__ == "__main__":
    sys.argv = ["training.py", "--save_root", "./data"]
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--data-path", type=str, default="./iclevr/", help="Dataset Path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Which device the training is on.")
    parser.add_argument('--log', default='logs_ddpm/', help='path to tensorboard log')

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train.")
    parser.add_argument("--start_from_epoch", type=int, default=0, help="Number of epochs to train.")
    parser.add_argument("--save_root", type=str, required=True, help="The path to save your data")
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--lr', default=0.0002, help='Learning rate')
    parser.add_argument('--c_dim', default=4, help='Condition dimension')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--outf_checkpoint', default='./checkpoint/DDPM/outf_checkpoint', help='Path to save model checkpoint')
    parser.add_argument("--outf", default="./checkpoint/DDPM/outf", help="folder to output images")

    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for training.")
    parser.add_argument('--timesteps', default=1000, help='Time step for diffusion')
    parser.add_argument('--test_file', default='test.json', help='Test file')
    parser.add_argument('--test_batch_size', default=32, help='Test batch size')
    parser.add_argument('--figure_file', default='figure/origin', help='Figure file')
    parser.add_argument('--resume', default=False, help='Continue for training')
    parser.add_argument('--model_path', default='ckpt', help='Path to save model checkpoint')
    parser.add_argument('--ckpt', default='net.pth', help='Checkpoint for network')

    args = parser.parse_args()

    cudnn.benchmark = True
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.outf, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    traindataset = iclevrDataset(mode='train', dataset_root='iclevr', json_root='json')
    traindataset_subset = Subset(traindataset, list(range(10)))

    train_dataloader = DataLoader(
        traindataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    testdataset = iclevrDataset(mode='test', dataset_root='iclevr', json_root='json')
    testdataset_subset = Subset(testdataset, list(range(10)))

    test_dataloader = DataLoader(
        testdataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    wandb.init(
        project = 'Deep Learning Lab6 Flexolk',
        config = {"batch_size":args.batch_size, 
                "epoch": 60, 
                "embedding": "nn.Linear", 
                "Type": "DDDDAA",
                "Block_size": 'bigger',
                "Resume": False
        },
        name = "Big block"
    )
    

    train_ddpm = TrainDDPM(args)

    if args.resume:
        train_ddpm.load_state_dict(torch.load(os.path.join(args.model_path, args.ckpt)))

    train_ddpm.train(train_dataloader, test_dataloader)
    
    wandb.finish()
