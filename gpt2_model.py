import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time
import math
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# simple launch (single GPU/CPU)
# python gpt2_model.py

# DDP launch (multi GPU)
# torchrun --standalone --nproc_per_node=8 gpt2_model.py

@dataclass
class GPT2Config:
    """
    Configuration for GPT-2 model.
    Note this is the 124M parameter version (not 1.5B).
    """
    vocab_size: int = 50257 # GPT-2 vocabulary size: 50000 BPE tokens + 256 bytes tokens + 1 <\endoftext> token
    # vocab_size: int = 50304 # round up to nearest multiple of 128
    block_size: int = 1024 # max sequence length
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension
    # dropout: float = 0.1 # dropout rate

class CausalSelfAttention(nn.Module):
    """GPT-2 Causal Self Attention"""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        # query, key, value projections in one batch, W_q, W_k, W_v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection, W_o
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape # B: batch size, T: sequence length, C: embedding dimension
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2) # (B, T, C), (B, T, C), (B, T, C)
        # split into multiple heads 
        # (B, T, C) -> (B, T, n_head, C // n_head) -> (B, n_head, T, C // n_head) -> (B, n_head, T, head_dim)
        head_dim = C // self.config.n_head
        q = q.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        # S = QK^T / sqrt(d_k)
        # attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5) # (B, n_head, T, T)
        # # causal mask : cannot look into the future tokens
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # (B, n_head, T, T)
        # # P = softmax(S)
        # attn = F.softmax(attn, dim=-1) # (B, n_head, T, T)
        # # O = PV
        # out = attn @ v # (B, n_head, T, head_dim)
        # Flash Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C) concat
        # output projection
        out = self.c_proj(out) # (B, T, C) W_o
        return out

class MLP(nn.Module):
    """GPT-2 MLP"""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x) # GELU activation https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
        x = self.c_proj(x)
        # x = self.dropout(x)
        return x

class Block(nn.Module):
    """GPT-2 Transformer Block"""
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    """GPT-2 Model"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # weight tying with wte
        # self.lm_head.weight = self.transformer.wte.weight 
        # TODO: why this will cause error? answer: init value mismatch (see README.md)
        # output: Hello, I'm a language model model model model ...
        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self.__init_weights)
    
    # 1/sqrt(768) = 0.036
    # TODO: what if use xavier init insead?
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = (2 * self.config.n_layer) ** -0.5 # 2 residual connections per transformer block
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, C)
        tok_emb = self.transformer.wte(idx) # (B, T, C)
        x = tok_emb + pos_emb # (B, T, C)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name: str) -> 'GPT2':
        """Load pre-trained GPT-2 model from Hugging Face"""
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from {model_name}...")
        
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_name]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # config_args['dropout'] = 0.1

        config = GPT2Config(**config_args)
        model = GPT2(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # skip the bias parameter in the attention layer

        # load weights from huggingface
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf = hf_model.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # skip the bias parameter in the attention layer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        # transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
        # transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
        # transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
        # transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
        transposed_keys = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']        
        # copy weights
        for k in sd_keys_hf:
            if any(k.endswith(tk) for tk in transposed_keys):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """Configure the optimizer"""
        # all parameters in the model
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # parameters that require gradients
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # parameters with weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # parameters without weight decay
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"Number of parameters with weight decay: {len(decay_params)}, with {num_decay_params,:} params")
        print(f"Number of parameters without weight decay: {len(no_decay_params)}, with {num_no_decay_params,:} params")
        # create optimizer
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # print(f"Fused AdamW available: {fused_available}")
        # used_fused = fused_available and device == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=(device == 'cuda'))
        return optimizer

import numpy as np
def load_tokens(filename: str):
    npt = np.load(filename)
    ptt = torch.from_numpy(npt).long()
    return ptt

import tiktoken
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0
        if master_process:
            print(f"Found {len(shards)} shards for {split} split")

        # state, init at shard zero
        # self.current_shard = 0
        # self.tokens = load_tokens(shards[self.current_shard])
        self.reset()

        # with open('input.txt', 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"Loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        # print(f"1 epoch = {len(self.tokens) // (B*T) * B * T} tokens")

        # self.current_pos = B * T * process_rank

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = B * T * self.process_rank

    def next_batch(self):
        """Get the next batch of data"""
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B*T*self.num_processes
        # if loading new batch is out of bound, advance to the next shard
        if self.current_pos + B*T*self.num_processes+1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = B * T * self.process_rank
        return x, y


# simple launch (single GPU/CPU)
# python gpt2_model.py

# DDP launch (multi GPU)
# torchrun --standalone --nproc_per_node=8 gpt2_model.py
if __name__ == "__main__":

    # # device check
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # else:
    #     device = 'cpu'
    # print(f"Using device: {device}")

    # DDP setup
    # torchrun cmd sets the env var RANK, LOCAL_RANK, WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0 # for logging, checkpointing, etc.
    else:
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # get a data batch
    # import tiktoken
    # enc = tiktoken.get_encoding('gpt2')
    # with open('input.txt', 'r') as f:
    #     text = f.read()
    # tokens = enc.encode(text)
    # B, T = 4, 32
    # tokens = torch.tensor(tokens[:B*T+1])
    # tokens = tokens.to(device)
    # x = tokens[:-1].view(B, T)
    # y = tokens[1:].view(B, T)

    # init data loader
    # train_loader = DataLoaderLite(B=4, T=32)
    total_batch_size = 524288 # 8 * 1024 * 64 = 2^19
    B, T = 8, 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by (B * T * ddp_world_size)"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"Total batch size: {total_batch_size}, effective batch size: {B * T * ddp_world_size}, gradient accumulation steps: {grad_accum_steps}")
        
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

    # torch.set_float32_matmul_precision('high')

    # init model
    model = GPT2(GPT2Config(vocab_size=50304))
    model.to(device)
    # logits, loss = model(x, y)
    # print(loss)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # learning rate scheduler: cosine decay with warmup
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    # warmup_steps = 10
    # max_steps = 50
    warmup_steps = 715 # 375M / 524288 = 715.25
    max_steps = 19073 # dataset_size / batch_size = 10B / 524288 = 19073.48
    def get_lr(it):
        # 1) linear warmup
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > max_steps, return min_lr
        if it > max_steps:
            return min_lr
        # 3) cosine decay between warmup_steps and max_steps down to min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = .5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coeff
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

    # log dir
    log_dir = 'logs'
    if master_process:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training.log')
        with open(log_file, 'w') as f:
            pass
            # f.write('step,val_loss,train_loss,lr,time,tokens_per_sec\n')
    
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # evaluating
        if step % 256 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: step {step:5d}, val loss: {val_loss_accum.item():.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} val,{val_loss_accum.item():.4f}\n")
                # save checkpoint
                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                    }
                    torch.save(checkpoint, os.path.join(log_dir, f'ckpt.{step:05d}.pt'))


        # training
        model.train()
        optimizer.zero_grad() # MUST clear gradients from previous step (default is to accumulate)
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device) 
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward() # compute gradients
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # gradient clipping: prevents exploding gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step() # update weights
        end.record()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000 # ms
        dt_event = start.elapsed_time(end) # ms
        tokens_process = B*T*grad_accum_steps*ddp_world_size
        tokens_per_sec = tokens_process / (t1 - t0)
        if master_process:
            print(f"step {step:5d}, loss: {loss_accum.item():.4f}, norm: {norm.item():.4f}, lr: {lr:.2e}, time: {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} train,{loss_accum.item():.4f}\n")

    if ddp:
        destroy_process_group()

    exit(0)

    # model = GPT2.from_pretrained('gpt2')
    model = GPT2(GPT2Config())
    model.eval()
    model.to(device)

    num_return_sequences = 5
    max_new_tokens = 30

    # tokenizer
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    prompt = "Hello, I'm a language model"
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device) # (num_return_sequences, T)
    x = tokens.to(device)

    # generate - autoregressive
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :] # (num_return_sequences, vocab_size)
            probs = F.softmax(logits, dim=-1) # (num_return_sequences, vocab_size)
            top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1)
            selected_indices = torch.multinomial(top_k_probs, num_samples=1)
            next_tokens = torch.gather(top_k_indices, dim=-1, index=selected_indices)
            x = torch.cat([x, next_tokens], dim=1)
    
    for step in range(num_return_sequences):
        tokens = x[step, :max_new_tokens].tolist()
        decoded = enc.decode(tokens)
        print(f">{step}: {decoded}")
    
    