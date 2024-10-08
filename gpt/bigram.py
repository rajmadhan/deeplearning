import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


os.chdir(os.path.dirname(__file__))

# hyper parameters
batch_size = 4
block_size = 8
n_embed = 64
# max_iters = 1000
# eval_interval = 100
# eval_iters = 25
max_iters = 10000
eval_interval = 500
eval_iters = 250
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx):
        # idx - B x T
        logits = self.token_embed_table(idx)
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
    

class BigramLanguageModel2(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embed_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=True)
    
    def forward(self, idx):
        # idx - B x T
        idx = idx[:, -block_size:]
        B, T = idx.shape
        token_emb = self.token_embed_table(idx)
        pos_emb = self.pos_embed_table(torch.arange(T, device=device))
        emb = token_emb + pos_emb
        logits = self.lm_head(emb)
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

for iter in range(max_iters):
    if iter%eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train'].item(): .4f}, val loss {losses['val'].item(): .4f}')

    x, y = get_batch('train')
    logits = model(x)
    loss = model.loss(logits, y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

torch.save(model, 'bigram.pt')
# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
