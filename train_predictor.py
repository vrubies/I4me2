import os
import time
import math
import pickle
import random
import json
import yfinance as yf
from datetime import datetime, timedelta
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, LLMForecaster

stock_list = ["MSFT", "AAPL", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "UNH", 
              "LLY", "JPM", "AVGO", "XOM", "V", "JNJ", "PG", "MA", "HD", "MRK", "COST", "ABBV", 
              "CVX", "ADBE", "CRM", "PEP", "KO", "BAC", "WMT", "AMD", "MCD", "ACN", "NFLX", 
              "CSCO", "TMO", "INTC", "LIN", "ABT", "WFC", "CMCSA", "PFE", "DIS", "INTU", "VZ", 
              "ORCL", "AMGN", "QCOM", "DHR", "TXN", "PM", "UNP", "IBM", "CAT", "COP", "SPGI", 
              "BA", "NOW", "GE", "HON", "NKE", "NEE", "AMAT", "GS", "T", "RTX", "LOW", "PLD", 
              "UBER", "BKNG", "MS", "UPS", "ISRG", "ELV", "MDT", "BLK", "AXP", "SBUX", "VRTX", 
              "DE", "BMY", "TJX", "GILD", "CVS", "C", "LMT", "AMT", "SCHW", "MDLZ", "SYK", "REGN", 
              "LRCX", "ADP", "PGR", "MMC", "ADI", "ETN", "CB", "MU", "PANW", "CI"]
print("Num tickers = ", len(stock_list))

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-predictor'
llm_dir = 'gpt2'
context_length_multiplier = 2
eval_interval = 2000
log_interval = 1
eval_iters = 10
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'fool_articles/dated_articles'
gradient_accumulation_steps = 1 # 5 * 8 # used to simulate larger batch sizes
batch_size = 3 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
# print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
# Load stock history from JSON file
with open('data/fool_articles/stock_history.json', 'r') as file:
    stock_history = json.load(file)

def split_data_folders(data_dir, val_percent=0.05):
    all_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    random.shuffle(all_folders)
    num_val = int(len(all_folders) * val_percent)
    val_folders = all_folders[:num_val]
    train_folders = all_folders[num_val:]
    return train_folders, val_folders
data_dir = os.path.join('data', dataset)
train_folders, val_folders = split_data_folders(data_dir)

def parse_date_from_folder(folder_name):
    # Assuming folder name is of the format "yyyy-mm-dd"
    return datetime.strptime(folder_name, "%Y-%m-%d")

def get_stock_returns(date, stock_list):
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=7)).strftime('%Y-%m-%d')
    returns = {}

    for stock in stock_list:
        start_price = stock_history.get(stock, {}).get(start_date, 0)
        end_price = stock_history.get(stock, {}).get(end_date, 0)

        if start_price and end_price:
            returns[stock] = (end_price - start_price) / start_price
        else:
            returns[stock] = 0.0

    return returns

def load_and_process_data(folder_path, max_length):
    file_path = os.path.join(folder_path, 'dated_data.bin')
    data = np.memmap(file_path, dtype=np.uint16, mode='r')

    if len(data) > max_length:
        start_index = random.randint(0, len(data) - max_length)
        return data[start_index:start_index + max_length]
    else:
        # Pad data if shorter than max_length
        padded_data = np.pad(data, (0, max_length - len(data)), 'constant')
        return padded_data

def get_batch(split):
    folders = train_folders if split == 'train' else val_folders
    x, y = [], []

    for _ in range(batch_size):
        selected_folder = random.choice(folders)
        date = parse_date_from_folder(os.path.basename(selected_folder))
        stock_returns = get_stock_returns(date, stock_list)

        article_data = load_and_process_data(selected_folder, block_size * context_length_multiplier)
        x.append(torch.from_numpy(article_data.astype(np.int64)))
        y.append(stock_returns)

    x = torch.stack(x)
    y = torch.tensor([list(returns.values()) for returns in y])  # Assuming returns are in the same order as stock_list

    # x, y = x.pin_memory().to('cuda', non_blocking=True), y.to('cpu')
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

x_train, y_train = get_batch('train')
print(x_train.shape, y_train.shape)
print(x_train[0], y_train[0])
print("Loader working")
# attempt to derive vocab_size from the dataset
# meta_path = os.path.join(data_dir, 'meta.pkl')
# meta_vocab_size = None
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']
#     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

print(f"Loading GPT model from {llm_dir}")
ckpt_path = os.path.join(llm_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location='cuda')
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = GPTConfig(**model_args)
model_A = GPT(gptconf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model_A.load_state_dict(state_dict)
model_A.eval()
model_A.to('cuda')
model_A = torch.compile(model_A)

# Freeze all the parameters so as to not change GPT-2 weights
for param in model_A.parameters():
    param.requires_grad = False

# Create the LLMForecaster model ================================================================================
model = LLMForecaster(model_A, context_length_multiplier, stock_list, gptconf)
model = torch.compile(model) # requires PyTorch 2.0
model.to(device)

# 1. Check Model Loading
print("First few parameters of GPT model:")
for name, param in model.named_parameters():
    print(name, param.size())
    break

# 2. Test Forward Pass of GPT Model
dummy_input = torch.randint(high=model_args['vocab_size'], size=(1, gptconf.block_size * 2)).to('cuda')
start_time = time.time()  # Start time measurement
with torch.no_grad():
    gpt_output = model(dummy_input)
end_time = time.time()  # End time measurement
print(f"Time taken for forward pass: {end_time - start_time:.4f} seconds")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint = None # free up memory

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    model_A.eval()
    return out

# helps estimate prediction accuracy
@torch.no_grad()
def estimate_profits():
    model.eval()
    reps = 3
    for split in ['val']:
        profits_predicted = torch.zeros(reps)
        profits_actual = torch.zeros(reps)
        for k in range(reps):
            pred_profit = 0.0
            actual_profit = 0.0
            X, Y = get_batch(split)
            with ctx:
                predictions = model(X)

            for i, p in enumerate(predictions):
                for j, pp in enumerate(p):
                    if pp > 0.05:
                        pred_profit += 100 * pp
                        actual_profit += 100 * Y[int(i)][int(j)]
            profits_actual[k] = actual_profit / reps
            profits_predicted[k] = pred_profit / reps
        print("A/P: ", profits_actual.mean().item() , " | ", profits_predicted.mean().item())
    model.train()
    model_A.eval()

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0
print("Starting training loop...")
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        estimate_profits()
        # if local_iter_num >= 5: # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
