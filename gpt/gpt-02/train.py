import torch
from model import GPT, GPTConfig
from dataloader import DataLoaderLite
from datetime import datetime
import time

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    B, T = 32, 128
    dataloader = DataLoaderLite("gpt/tiny-shakespeare.txt", B, T)
    #model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    start_time = time.time()
    for i in range(500):
        optimizer.zero_grad()
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step: {i}, loss: {loss.item(): .6f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time: {elapsed_time: .3f}")
