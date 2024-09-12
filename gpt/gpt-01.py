import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from datetime import datetime

os.chdir(os.path.dirname(__file__))

# hyper parameters
batch_size = 64
block_size = 256
n_embed = 384
num_head = 6
n_decoder_blocks = 6
max_iters = 5000
eval_interval = 500
eval_iters = 100
learning_rate = 3e-4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# batch_size = 64
# block_size = 256
# n_embed = 384
# num_head = 6
# n_decoder_blocks = 6
# max_iters = 1000
# eval_interval = 100
# eval_iters = 100
# learning_rate = 3e-4
# dropout = 0.2

# batch_size = 64
# block_size = 256
# n_embed = 384
# num_head = 1
# n_decoder_blocks = 1
# max_iters = 1000
# eval_interval = 100
# eval_iters = 250
# learning_rate = 1e-3

torch.manual_seed(1337)

text = open('tiny-shakespeare.txt', 'r', encoding='utf-8').read()
print('dataset len: ', len(text))
print(text[:1000])

chars = sorted(list(set(c for c in text)))
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
vocab_size = len(chars)
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long).to(device)
train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]
print('train size: ', len(train_data))
print('val size: ', len(val_data))

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SelfAttentionHead(nn.Module):
    def __init__(self, n_embed, headsize) -> None:
        super().__init__()
        self.query = nn.Linear(n_embed, headsize, bias=False)
        self.key = nn.Linear(n_embed, headsize, bias=False)
        self.value = nn.Linear(n_embed, headsize, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x) -> torch.tensor: # x: 
        B, T, C = x.shape
        q = self.query(x) # 
        k = self.key(x) #         
        wt = q @ k.transpose(-1, -2) * C**-0.5 # BTT
        wt = wt.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wt = F.softmax(wt, dim=-1) # BTT
        wt = self.dropout(wt)
        v = self.value(x) # 
        out = wt @ v # 
        return out

class SelfAttentionMultiheadVer1(nn.Module):
    def __init__(self, n_embed, num_head):
        super().__init__()
        head_size = n_embed // num_head
        self.heads = nn.ModuleList([SelfAttentionHead(n_embed, head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class SelfAttentionMultihead(nn.Module):
    def __init__(self, n_embed, num_head) -> None:
        super().__init__()
        self.c_attn = nn.Linear(n_embed, 3*n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_embed = n_embed
        self.n_head = num_head
    
    def forward(self, x) -> torch.tensor: # x: 
        B, T, C = x.shape # C == n_embed
        attn = self.c_attn(x) # B, T, 3C
        q, k, v = attn.split(self.n_embed, dim=-1) # B, T, C
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B, nh, T, hs(head size)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B, nh, T, hs 
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B, nh, T, hs 
        wt = q @ k.transpose(-1, -2) * k.size(-1)**-0.5 # B, nh, T, T
        wt = wt.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wt = F.softmax(wt, dim=-1) # BTT
        wt = self.attn_dropout(wt)
        out = wt @ v # B, nh, T, hs
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out
   
class FeedForward(nn.Module):
    def __init__(self, fanin, middle, fanout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fanin, middle, bias=True),
            nn.ReLU(),
            nn.Linear(middle, fanout, bias=True),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        out = self.net(x)
        return out

class MLP(nn.Module):
    def __init__(self, fanin, middle, fanout):
        super().__init__()
        self.fc = nn.Linear(fanin, middle)
        self.proj = nn.Linear(middle, fanout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = new_gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, n_embed, num_head):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(n_embed)
        self.attn = SelfAttentionMultihead(n_embed, num_head)        
        self.layernorm2 = nn.LayerNorm(n_embed)
        #self.ffwd = FeedForward(n_embed, 4*n_embed, n_embed)
        self.mlp = MLP(n_embed, 4*n_embed, n_embed)
    
    def forward(self, x):
        x = x + self.attn(self.layernorm1(x))
        #x = x + self.ffwd(self.layernorm2(x))
        x = x + self.mlp(self.layernorm2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embed_table = nn.Embedding(block_size, n_embed)
        self.decoder_block = nn.Sequential(*[DecoderBlock(n_embed, num_head) for _ in range(n_decoder_blocks)])
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=True)
    
    def forward(self, idx):
        # idx - B x T
        idx = idx[:, -block_size:]
        B, T = idx.shape
        token_emb = self.token_embed_table(idx)
        pos_emb = self.pos_embed_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.decoder_block(x)
        logits = self.lm_head(x)
        return logits
    
    def loss(self, logits, targets):
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), reduction='mean')
        return loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:,-1,:]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1, replacement=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# load data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)
            loss = model.loss(logits, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel(vocab_size).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

print('# parameters: ', sum(p.numel() for p in model.parameters())/1e6, 'M')

start_time = time.time()
for iter in range(max_iters):
    if iter%eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        end_time = time.time()
        elapsed_time = end_time - start_time
        start_time = end_time
        print(f'step {iter}: train loss {losses['train'].item(): .4f}, val loss {losses['val'].item(): .4f}, time {elapsed_time: .3f}')

    x, y = get_batch('train')
    logits = model(x)
    loss = model.loss(logits, y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

torch.save(model, 'gpt-01-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.pt')
# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
#print(decode(model.generate(context, 500)[0].tolist()))
