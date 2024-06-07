import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel


class ConditionedUNet(nn.Module):
    def __init__(self, args, num_class = 24, embed_size = 512):
        super().__init__() 
        
        self.model = UNet2DModel(
            sample_size = 64,
            in_channels =  3,
            out_channels = 3,
            layers_per_block = 2,
            class_embed_type = None,
            block_out_channels = (128, 256, 256, 512, 512, 1024),
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",    
                "DownBlock2D",    
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D", 
            ), 
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",         
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        self.model.class_embedding = nn.Linear(num_class, embed_size)
    
    def forward(self, x, t, y):
        return self.model(x, t, class_labels = y).sample 

def corrupt(x, amount):
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x)
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1-amount) + noise*amount 
