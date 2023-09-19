import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange






class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

    
    
class fusion_defender(nn.Module):
    def __init__(self, t, h, w, channels, patch_t, patch_h, patch_w, dim, depth, heads, mlp_dim, 
                dropout, dim_head = 64):
        
        super().__init__()
        
        self.T = t
        self.H = h
        self.W = w

        self.channels = channels

        self.t = patch_t
        self.h = patch_h
        self.w = patch_w

        self.nt = self.T // self.t
        self.nh = self.H // self.h
        self.nw = self.W // self.w
        
        num_patches = self.nh * self.nw
        tubelet_dim = self.h * self.w * self.channels
        
        print(num_patches)
        print(tubelet_dim)
        
        self.linear_projection_img = nn.Linear(tubelet_dim, dim)
        self.linear_projection_im_rev = nn.Linear(dim, tubelet_dim)
        
        self.rearrange_split = Rearrange('b c (h ph) (w pw) ->b (h w) (c ph pw)', ph=28, pw=28)
        
        self.rearrange_return = Rearrange('b (h w) (c ph pw) ->b c (h ph) (w pw) ',h=8,w=8, ph=28, pw=28)
        
        self.pos_embedding_img = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        
        self.linear_projection_aud = nn.Linear(400, dim)
        self.linear_projection_aud_rev = nn.Linear(dim, 400)
        
        self.rearrange_split_aud = Rearrange('b  (h ph) (w pw) ->b (  h w) ( ph pw)', ph=20, pw=20)
        self.rearrange_return_aud = Rearrange('b ( h w) ( ph pw) ->b   (h ph) (w pw) ',h=4,w=15, ph=20, pw=20)
        
        self.pos_embedding_aud = nn.Parameter(torch.randn(1, 60 + 1, dim))
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
         
    def forward(self, imgs, audio):
        
        imgs = imgs
        audio = audio
        
        imgs = self.rearrange_split(imgs)
        
        imgs = self.linear_projection_img(imgs)
        
        imgs += self.pos_embedding_img[:, :-1]
    
        
        
        audio = self.rearrange_split_aud(audio)
        
        audio = self.linear_projection_aud(audio)
        
        audio += self.pos_embedding_aud[:, :-1]
        
        
        cat = torch.cat((imgs, audio),1)
        
        
        x = self.transformer(cat)
        
        imgs = x[:,:64,:]
        audio = x[:,64:,:]
        
        imgs = self.linear_projection_im_rev(imgs)
        rec_imgs = self.rearrange_return(imgs)
        
        audio = self.linear_projection_aud_rev(audio)
        rec_audio = self.rearrange_return_aud(audio)
        
        
        return [rec_imgs, rec_audio]    