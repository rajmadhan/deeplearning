import torch
import tiktoken

class DataLoaderLite:
    def __init__(self, input_file, B, T):
        self.B = B
        self.T = T

        with open(input_file, 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens // (B*T))} batches")

        self.curr_pos = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        if self.curr_pos + B*T + 1 > len(self.tokens):
            self.curr_pos = 0
        buf = self.tokens[self.curr_pos:self.curr_pos+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.curr_pos += B*T
        return x, y