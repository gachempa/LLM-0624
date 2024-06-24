# from Anfrej Karpathy's https://www.youtube.com/watch?v=l8pRSuU81PU&t=15s

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):

    def __init__(self,config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head ==0
        # k, q, v prjections for all heads but in a batch 
        self.c_attn=nn.Linear(config.n_embd, 3*config.n_embd)
        # outout projection
        self.c_proj=nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head=config.n_head
        self.n_embd=config.n_embd
        # not really a bias but a mask, following HI naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size,config.block_size))

    def forward(self,x):
        B,T,C=x.size() # batch size, sequence length, embedding dimensionality (n_embd) 
        # calculate q, k, v for all heads in batch and move head forward to be the batch ..
        # nh is "no. of heads", hs is head size, and C (# of channels)=nh*hs
        # eg in GPT-2 (124M), n_head=64, hs=64, so nh*hs=768 channels in the Xmer
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd, dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transport(1,2) # (B,nh,T,hs)
        q=q.view(B,T,self.n_head,C//self.n_head).transport(1,2) # (B,nh,T,hs)  
        v=v.view(B,T,self.n_head,C//self.n_head).transport(1,2) # (B,nh,T,hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        att=F.softmax(att,dim=-1)
        y=att@v # (B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs)
        y=y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side by side
        # output projection
        y=self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self,config) -> None:
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self,config) -> None:
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2)
        return x

@dataclass
class GPTConfig:
    block_size: int=1024 # max sequence length
    vocab_size: int=50527 # no. of tokens: 50,000 BPE merges + 256 bu=ytes tokens +1 <|endoft..
    n_layer: int=12
    n_head: int=12
    n_embd: int=768

class GPT(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)