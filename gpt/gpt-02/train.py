import torch
from model import GPT, GPTConfig
from dataloader import DataLoaderLite
from datetime import datetime
import time
import lr

if __name__ == "__main__":
    torch.manual_seed(42)    

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = "cuda"
    torch.set_float32_matmul_precision("high")

    total_batch_size = 524288
    B, T = 8, 1024
    assert total_batch_size % (B * T) == 0
    grad_accum_step = total_batch_size // (B * T)
    print(f"batch size: {B}, tokens: {T}, grad_accum_step: {grad_accum_step}")
    dataloader = DataLoaderLite("gpt/tiny-shakespeare.txt", B, T)
    #model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(block_size=T))
    model.to(device)
    t0 = time.time()
    # model = torch.compile(model)
    t1 = time.time()
    td = (t1 - t0)*1000
    print(f"model compile time: {td: .2f}ms")
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type=device)
    
    start_time = time.time()
    for step in range(100):
        t0 = time.time()
        optimizer.zero_grad()
        for g in optimizer.param_groups:
            g['lr'] = lr.get_lr(step)
        loss_accum = 0
        for micro_step in range(grad_accum_step):
            x, y = dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)                
            loss = loss / grad_accum_step
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        td = (t1 - t0)*1000
        tokens_per_sec = (dataloader.B * dataloader.T * grad_accum_step) / (t1 - t0)
        print(f"step: {step} | lr: {lr.get_lr(step): .6f} | loss: {loss_accum.item(): .6f} | norm: {norm: .4f} | td: {td: .2f}ms | tok/sec: {tokens_per_sec:.2f}")
        if not step%100:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"elapsed time: {elapsed_time: .3f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time: {elapsed_time: .3f}")
