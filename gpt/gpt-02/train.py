import torch
from model import GPT, GPTConfig
from dataloader import DataLoaderLite
from datetime import datetime
import time

if __name__ == "__main__":
    torch.manual_seed(42)    

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = "cuda"
    torch.set_float32_matmul_precision("high")

    B, T = 8, 1024
    dataloader = DataLoaderLite("gpt/tiny-shakespeare.txt", B, T)
    #model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(block_size=T))
    model.to(device)
    t0 = time.time()
    model = torch.compile(model)
    t1 = time.time()
    td = (t1 - t0)*1000
    print(f"model compile time: {td: .2f}ms")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

    start_time = time.time()
    for i in range(100):
        t0 = time.time()
        optimizer.zero_grad()
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        td = (t1 - t0)*1000
        tokens_per_sec = (dataloader.B * dataloader.T) / (t1 - t0)
        print(f"step: {i} | loss: {loss.item(): .6f} | norm: {norm: .4f} | td: {td: .2f}ms | tok/sec: {tokens_per_sec:.2f}")
        if not i%100:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"elapsed time: {elapsed_time: .3f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time: {elapsed_time: .3f}")
