# %%
import argparse
import sys

import torch
from Trainer import VAE_Model
import matplotlib.pyplot as plt

def plot(model):
    all_loss = model.all_loss
    all_loss = torch.tensor(all_loss, device = 'cpu')
   
    all_teacher_forcing_ratio = model.all_teacher_forcing_ratio
    all_teacher_forcing_ratio = torch.tensor(all_teacher_forcing_ratio, device = 'cpu')

    fig, ax1 = plt.subplots()
    ax1.plot(all_loss, label='Model Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Model Loss')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(all_teacher_forcing_ratio, label='Teacher Forcing Ratio', color='red')
    ax2.set_ylabel('Teacher Forcing Ratio')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Without KL annealing')

    fig.tight_layout()
    plt.show()

    """     torch.save({
        "state_dict": model.state_dict(),
        "optimizer": model.state_dict(),  
        "lr"        : model.scheduler.get_last_lr()[0],
        "tfr"       :   model.tfr,
        "last_epoch": model.current_epoch,
        "all_loss": model.all_loss,
        "all_teacher_forcing_ratio": all_teacher_forcing_ratio,
    }, './data/test.pth') """

    pass

def main(args):
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    plot(model)

if __name__ == '__main__':
    sys.argv = ['Trainer.py', '--DR', '../LAB4_Dataset', '--save_root', './data', '--ckpt_path' , './data-new-without/epoch=6.ckpt' ]
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=1,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=5,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="Cyclical, Monotonic, WithoutKLAnnealing")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    args = parser.parse_args()
    main(args)

# %%
