# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
import textgrid
from tqdm import tqdm
import time
from contextlib import nullcontext
from pathlib import Path
import shutil
import math
import random
import random

# ML
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import pandas
import wandb
from einops import rearrange, reduce, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import schedulefree
from torch.profiler import profile, record_function, ProfilerActivity

# Local
from supervoice_valle import SupervoceNARModel, Tokenizer
from train.dataset import load_sampler, create_async_loader

# We speculate that original paper has about 6k tokens per GPU
# 6k tokens is routhly 3 rows, because a single row is a 1500-2500 tokens
# We have MUCH faster GPUs and therefore instead of gradient accumulation,
# we increase batch size 4x and reduce number of gradients to just 4x
train_grad_accum_every = 8
train_batch_size = 8

# We speculate that learning rate is given for all GPUs, so we divide it by number of GPUs
train_lr_start = 1e-12
train_lr_max = 1e-5
train_steps = 600000
train_warmup_steps = 32000 # I am using faster warmup - it is more natural for me after working on voicebox
train_schedule_free = False

train_loader_workers = 32
train_log_every = 1
train_save_every = 1000
train_watch_every = 1000
train_evaluate_every = 200
train_evaluate_batches = 10
train_mixed_precision = "fp16" # "bf16" or "fp16" or None
train_clip_grad_norm = 0.2 # Common reproductions are using 100, but i am usually use 0.2
train_compile = False

# Train
def main():
    print("PyTorch: ", torch.__version__)

    # Prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every, mixed_precision=train_mixed_precision)
    device = accelerator.device
    dtype = torch.float16 if train_mixed_precision == "fp16" else (torch.bfloat16 if train_mixed_precision == "bf16" else torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    lr_start = train_lr_start * accelerator.num_processes
    lr_max = train_lr_max * accelerator.num_processes

    # Prepare dataset
    accelerator.print("Loading dataset...")
    tokenizer = Tokenizer("./tokenizer_text.model")
    train_sampler = load_sampler("./external_datasets/libriheavy/libriheavy_cuts_small.jsonl.gz", "./external_datasets/libriheavy-encodec/", train_batch_size, tokenizer)
    train_loader = create_async_loader(train_sampler, num_workers = train_loader_workers)
    train_cycle = cycle(train_loader)

    # Model
    accelerator.print("Loading model...")
    step = 1
    model = SupervoceNARModel().to(device)
    raw_model = model
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    if not train_schedule_free:
        optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], train_lr_start, betas=[0.9, 0.95],weight_decay=0.01, eps=1e-6)
    else:
        optim = schedulefree.AdamWScheduleFree([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], lr=train_lr_max, betas=[0.9, 0.95],weight_decay=0.01, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)
    if train_compile:
        model = torch.compile(model, mode="reduce-overhead")
    model, optim = accelerator.prepare(model, optim)

    # torch.cuda.set_sync_debug_mode("error")

    # Train step
    def train_step():
        model.train()
        if train_schedule_free:
            optim.train()

        # Update LR
        if not train_schedule_free:
            if step < train_warmup_steps:
                lr = (lr_start + ((lr_max - lr_start) * step) / train_warmup_steps)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                lr = lr / accelerator.num_processes
            else:
                scheduler.step()
                lr = scheduler.get_last_lr()[0] / accelerator.num_processes
        else:
            lr = lr_max / accelerator.num_processes

        # Load batch
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if accelerator.is_main_process else [], profile_memory=True, record_shapes=True) as prof:
            for _ in range(train_grad_accum_every):
                with accelerator.accumulate(model):
                    with accelerator.autocast():

                        # Load Batch
                        with record_function("load_batch"):

                            # Load batch
                            audio, text = next(train_cycle)

                            # Split audio
                            texts = []
                            audio_full = []
                            audio_partial = []
                            audio_codecs = []
                            for B in range(len(audio)):
                                a = audio[B].squeeze(0)
                                t = text[B].squeeze(0)
                                audio_duration = a.shape[1]
                                min_duration = 75 * 3
                                max_duration = audio_duration // 2 
                                if max_duration > min_duration:
                                    audio_split = random.randint(min_duration, max_duration)
                                else:
                                    audio_split = max_duration
                                with record_function("load_batch:append"):
                                    audio_full.append(a[:, :audio_split].to(device, non_blocking=True))
                                    audio_partial.append(a[:, audio_split:].to(device, non_blocking=True))
                                    audio_codecs.append(random.randint(1, 7))
                                    texts.append(t.to(device, non_blocking=True))

                        # Forward
                        with record_function("forward"):
                            _, loss = model(
                                condition_text = texts,
                                condition_audio = audio_full,
                                audio = audio_partial,
                                codec = audio_codecs,
                                loss = True
                            )
                    
                            # # Check if loss is NaN
                            # if torch.isnan(loss):
                            #     raise ValueError("Loss is NaN")
                        
                        # Backprop
                        with record_function("backward"):
                            optim.zero_grad()
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(model.parameters(), train_clip_grad_norm)
                            optim.step()

                            # Log skipping step
                            if not train_schedule_free:
                                if optim.step_was_skipped:
                                    accelerator.print("Step was skipped")
        
        if accelerator.is_main_process:
            prof.export_chrome_trace("trace.json")
            print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=50))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        
        return 0, lr

    #
    # Do step
    #

    train_step()


#
# Utility
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    

if __name__ == "__main__":
    main()