import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])
        
    def load_checkpoint(self, load_ckpt_path):
        ckpt = torch.load(load_ckpt_path)
        self.load_state_dict(ckpt['state_dict'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        # quant_z, indices, _ = self.vqgan.encode(x)
        codebook_mapping, codebook_indices, q_loss = self.vqgan.encode(x)
        quant_z = codebook_mapping
        indices = codebook_indices
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        
        z_indices=None #ground truth
        logits = None  #transformer predict the probability of tokens

        # get the codebook z
        _, z_indices = self.encode_to_z(x) 
        
        sample_ratio = self.gamma(np.random.uniform()) 
        N = z_indices.shape[1]
        r = math.floor(sample_ratio * N)
        sample = torch.rand(z_indices.shape, device=z_indices.device) #隨機生成跟 z 一樣維度的陣列
        sample = sample.topk(r, dim=1).indices # 隨機取得 r 個 mask 的索引
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)  #生成跟 z 一樣維度的陣列, 填充0
        mask.scatter_(dim=1, index=sample, value=True) # 選取到的索引填入到 mask

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device) # 10*256 全是 1024
        a_indices = (mask * z_indices) + ((~mask) * masked_indices) # masked 的地方會變成 1024
        target = z_indices
        logits = self.transformer(a_indices) #然後把 masked 過後的 z 放到 bidirectional transforrmer

        return logits, target
    
    def create_input_tokens_normal(self, num, label=None):
        blank_tokens = torch.ones((num, self.num_image_tokens), device="cuda")
        masked_tokens = self.mask_token_id * blank_tokens
        return masked_tokens.to(torch.int64)
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, ratio, mask_num):
        # 把 True 的地方設為 1024,  也就是把 masked 的 position 設為 1024
        z_indices[mask_bc] = self.mask_token_id

        logits = self.transformer(z_indices)
        
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = nn.functional.softmax(logits, -1) # (bs, 256)

        #FIND MAX probability for each token value
        z_indices_predict_prob, logits_max_indices = torch.max(logits, -1) #找到所有 token 之中的最大值
        z_indices_predict_prob[~mask_bc] = float('inf') #把 mask 之外的 token 的機率設為無限大

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        _, sorted_indices = torch.sort(confidence)
        logits_max_indices[~mask_bc] = z_indices[~mask_bc]
        mask_bc[:, sorted_indices[:, math.floor(ratio*mask_num):]] = False
        
        return logits_max_indices, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
