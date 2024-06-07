import os
import random
import sys
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from matplotlib import transforms
from torchvision import utils as vutils
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.tensorboard import SummaryWriter

from dataset import iclevrDataset
from dcgan import weights_init, Generator, Discriminator
from evaluator import evaluation_model

class TrainDCGAN:
    def __init__(self, args):
        self.args = args
        self.current_epoch = 0

        self.nc = int(args.nc)
        self.nz = int(args.nz)
        self.ngf = int(args.ngf)
        self.ndf = int(args.ndf)

        self.netG = Generator(args, self.nc, self.nz, self.ngf).to(args.device)
        self.netG.apply(weights_init)
        if args.netG != "":
            self.netG.load_state_dict(torch.load(args.netG))
        print(self.netG)

        self.netD = Discriminator(args, self.nc, self.ndf).to(args.device)
        self.netD.apply(weights_init)
        if args.netD != "":
            self.netD.load_state_dict(torch.load(args.netD))
        print(self.netD)

        self.writer = SummaryWriter(args.log)
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))        
        
        if args.dry_run:
            args.epochs = 1

        self.prepare_training()
        self.G_losses = []
        self.D_losses = []

    @staticmethod
    def prepare_training():
        pass

    def train(self, train_loader, val_loader):
        for epoch in range(self.args.start_from_epoch + 1, self.args.epochs + 1):
            self.current_epoch = epoch
            self.train_one_epoch(epoch, train_loader)
            avg_acc = self.eval_one_epoch(epoch, val_loader)
            self.save(avg_acc)

    def train_one_epoch(self, epoch, train_loader):
        iters = 0
        total_loss_d = 0
        total_loss_g = 0
        self.netG.train()
        self.netD.train()
        for i, (image, cond) in enumerate(pbar := tqdm(train_loader)):
            # batch_size = 64, channels = 3, width = 64, height = 64

            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()

            device = args.device
            real_image = image.to(device)
            cond = cond.to(device)
            batch_size = image.size(0)
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            real_label = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(device)
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake_image = self.netG(noise, cond)
            fake_label = ((0.3 - 0.0) * torch.rand(batch_size) + 0.0).to(device)
            if random.random() < 0.1:
                real_label, fake_label = fake_label, real_label
            output = self.netD(real_image, cond)
            D_x = output.mean().item()
            errD_real = self.criterion(output, real_label)
            output = self.netD(fake_image.detach(), cond)
            errD_fake = self.criterion(output, fake_label)
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            errD.backward()
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator_label = torch.ones(batch_size).to(device)  # fake labels are real for generator cost
            output = self.netD(fake_image, cond)
            errG = self.criterion(output, generator_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizerG.step()

            self.writer.add_scalar('Train Step/Loss D', errD.item(), iters)
            self.writer.add_scalar('Train Step/Loss G', errG.item(), iters)
            self.writer.add_scalar('Train Step/D(x)', D_x, iters)
            self.writer.add_scalar('Train Step/D(G(z))', D_G_z1, iters)
            pbar.set_description('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, args.epochs, i, len(train_dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            pbar.refresh()
            total_loss_d += errD.item()
            total_loss_g += errG.item()
            if iters % 100 == 0:
                noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                vutils.save_image(real_image,
                    '%s/real_samples.png' % args.outf,
                    normalize=True)
                vutils.save_image(fake_image.detach(),
                    '%s/fake_samples.png' % args.outf,
                    normalize=True)
            iters += 1
        self.writer.add_scalar('Train Epoch/Loss D', total_loss_d/len(train_dataloader), epoch)
        self.writer.add_scalar('Train Epoch/Loss G', total_loss_g/len(train_dataloader), epoch)

    
    def tqdm_bar_train(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    @torch.no_grad()
    def eval_one_epoch(self, epoch, test_dataloader):
        args = self.args
        evaluator = evaluation_model()
        self.netG.eval()
        self.netD.eval()
        avg_acc = 0
        with torch.no_grad():
            for sample in range(10):
                for i, cond in enumerate(test_dataloader):
                    cond = cond.to(args.device)
                    batch_size = cond.size(0)
                    noise = torch.randn(batch_size, args.nz, 1, 1, device=args.device)
                    fake_image = self.netG(noise, cond)
                    vutils.save_image(fake_image.detach(),
                        '%s/epoch_%d_fake_test_sample_%d.png' % ( args.outf, epoch, sample),
                        normalize=True)
                    acc = evaluator.eval(fake_image, cond)
                print(f'Sample {sample+1}: {acc*100:.2f}%')
                avg_acc += acc
            avg_acc /= 10
            print(f'Average acc: {avg_acc*100:.2f}%')
        return avg_acc

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, avg_acc):
        torch.save(
            {
                "state_dict": self.netG.state_dict(),
                "lr": self.args.lr,
                "last_epoch": self.current_epoch,
                "avg_acc": avg_acc,
            },
            f"{self.args.outf_checkpoint}/netG_epoch_{self.current_epoch}_{avg_acc}.pth",
        )
        torch.save(
            {
                "state_dict": self.netD.state_dict(),
                "lr": self.args.lr,
                "last_epoch": self.current_epoch,
                "avg_acc": avg_acc,
            },
            f"{self.args.outf_checkpoint}/netD_epoch_{self.current_epoch}_{avg_acc}.pth",
        )
        print(f"save ckpt")

class TrainDDPM:
    pass

def collate_fn(batch):
    images, labels = zip(*batch)
    
    # Stack images (they should all be the same size)
    images = torch.stack(images)
    
    # Pad labels to the same length
    labels = [torch.as_tensor(label) for label in labels]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return images, labels_padded

if __name__ == "__main__":
    sys.argv = ["training.py", "--save_root", "./data"]
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--data-path", type=str, default="./iclevr/", help="Dataset Path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Which device the training is on.")
    parser.add_argument('--log', default='logs_dqgan/', help='path to tensorboard log')

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train.")
    parser.add_argument("--start_from_epoch", type=int, default=0, help="Number of epochs to train.")
    parser.add_argument("--save_root", type=str, required=True, help="The path to save your data")
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--nc", type=int, default=100, help="number of condition embedding dim")
    parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
    parser.add_argument("--ngf", type=int, default=300)
    parser.add_argument("--ndf", type=int, default=100)
    parser.add_argument("--netG", default="", help="path to netG (to continue training)")
    parser.add_argument("--netD", default="", help="path to netD (to continue training)")
    parser.add_argument("--outf_checkpoint", default="./checkpoint/DCGAN/outf_checkpoint", help="model checkpoints")
    parser.add_argument("--outf", default="./checkpoint/DCGAN/outf", help="folder to output images")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate, default=0.0002")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")

    args = parser.parse_args()

    cudnn.benchmark = True
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.outf, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataloader = DataLoader(
        iclevrDataset(mode='train', dataset_root='iclevr', json_root='json'),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        iclevrDataset(mode='test', dataset_root='iclevr', json_root='json'),
        batch_size=args.batch_size,
        shuffle=False
    )

    train_dcgan = TrainDCGAN(args)
    train_dcgan.train(train_dataloader, test_dataloader)
