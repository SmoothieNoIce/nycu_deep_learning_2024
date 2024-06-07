import torch.nn as nn
import torch
import math

#https://github.com/dome272/MaskGIT-pytorch/tree/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8

class Attention(nn.Module):
    def __init__(self, dim=768, heads=8, dropout_p=0.1):
        super(Attention, self).__init__()
        
        # 將輸入的維度經過線性轉換為 16 dim, 各個 head 有 16 dim
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        
        #減少過擬合
        self.norm = d ** 0.5
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attention_weights = q @ torch.transpose(k, 1, 2)
        attention_weights = attention_weights / self.norm
        attention_weights = torch.softmax(attention_weights, dim=1)
        attention_weights = self.dropout(attention_weights)
        attn = torch.matmul(attention_weights, v)
        return attn

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.self_attention_heads = nn.ModuleList([Attention(dim, num_heads) for _ in range(num_heads)])
        self.projector = nn.Linear(dim, dim)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(x)
            else:
                out = torch.cat((out, sa_head(x)), axis=-1)
        out = self.projector(out)
        return out

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    