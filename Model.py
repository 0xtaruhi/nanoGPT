import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

# Config class for Transformer model parameters
@dataclass
class Config:
    n_vocab: int
    d_model: int
    n_block: int
    n_head: int
    n_layer: int
    d_inner: int
    dropout: float
    bias: bool = False
    
    def __post_init__(self):
        self.d_k = self.d_v = self.d_model // self.n_head
        self.n_embd = self.d_model

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3*config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.proj_drop = nn.Dropout(config.dropout)
        
        self.dropout_p = config.dropout
        
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_k = config.d_k
        
    def forward(self, x):
        B, T, C = x.shape
        
        q,k,v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.train else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.proj_drop(self.proj(y))
        
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model,  config.d_inner, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
        
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, dim)
        self.word_embeddings = nn.Embedding(config.n_vocab, config.d_model)
        self.position_embeddings = nn.Embedding(config.n_block, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.layernorm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.n_vocab, bias=False)
        
        self.lm_head.weight = self.word_embeddings.weight
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, idx, targets=None):
        tok_emb = self.word_embeddings(idx)
        pos_emb = self.position_embeddings(torch.arange(idx.shape[1], device=idx.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)

        x = self.layernorm(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:,[-1],:])
            loss = None
            
        return logits, loss
        
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
            
        return idx