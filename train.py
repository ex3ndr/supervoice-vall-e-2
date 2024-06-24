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

# ML
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import pandas
import wandb
from einops import rearrange, reduce, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# Local
from supervoice_valle import SupervoceNARModel, Tokenizer
from train.dataset import load_sampler, create_async_loader

# Train parameters
train_experiment = "valle-08"
train_project="supervoice-valle"
train_auto_resume = True
train_grad_accum_every = 8 # Simulate 16 gpus using 2 gpus
train_steps = 600000
train_loader_workers = 32
train_batch_size = 8
train_log_every = 1
train_save_every = 1000
train_watch_every = 1000
train_evaluate_every = 200
train_evaluate_batches = 10
train_lr_start = 1e-7
train_lr_max = 5e-4
train_warmup_steps = 32000
train_mixed_precision = "fp16" # "bf16" or "fp16" or None
train_clip_grad_norm = 100
train_compile = False

# Train
def main():

    # Prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every, mixed_precision=train_mixed_precision)
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if train_mixed_precision == "fp16" else (torch.bfloat16 if train_mixed_precision == "bf16" else torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    lr_start = train_lr_start * accelerator.num_processes
    lr_max = train_lr_max * accelerator.num_processes

    # Prepare dataset
    accelerator.print("Loading dataset...")
    tokenizer = Tokenizer("./tokenizer_text.model")
    # train_sampler = load_sampler("./external_datasets/libriheavy/libriheavy_cuts_medium.jsonl.gz", "./external_datasets/libriheavy-medium-encodec/", train_batch_size, tokenizer)
    train_sampler = load_sampler("./external_datasets/libriheavy/libriheavy_cuts_small.jsonl.gz", "./external_datasets/libriheavy-encodec/", train_batch_size, tokenizer)
    train_loader = create_async_loader(train_sampler, num_workers = train_loader_workers)
    train_cycle = cycle(train_loader)

    # Model
    accelerator.print("Loading model...")
    step = 0
    model = SupervoceNARModel().to(device)
    raw_model = model
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], train_lr_start, betas=[0.9, 0.99],weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)
    if train_compile:
        model = torch.compile(model)
    
    # Accelerate
    model, optim = accelerator.prepare(model, optim)
    hps = {
        "train_lr_start": train_lr_start, 
        "train_lr_max": train_lr_max, 
        "grad_accum_every": train_grad_accum_every,
        "steps": train_steps, 
        "warmup_steps": train_warmup_steps,
        "mixed_precision": train_mixed_precision,
        "clip_grad_norm": train_clip_grad_norm,
    }
    accelerator.init_trackers(train_project, config=hps)
    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=train_watch_every * train_grad_accum_every)

    # Save
    def save():
        # Save step checkpoint
        fname = str(output_dir / f"{train_experiment}.pt")
        fname_step = str(output_dir / f"{train_experiment}.{step}.pt")
        torch.save({

            # Model
            'model': raw_model.state_dict(), 

            # Optimizer
            'optimizer': optim.state_dict(), 
            'scheduler': scheduler.state_dict(),
            'step': step 

        },  fname_step)

        # Overwrite main checkpoint
        shutil.copyfile(fname_step, fname)

    # Load
    if train_auto_resume and (output_dir / f"{train_experiment}.pt").exists():
        accelerator.print("Resuming training...")
        checkpoint = torch.load(str(output_dir / f"{train_experiment}.pt"), map_location="cpu")

         # Model
        raw_model.load_state_dict(checkpoint['model'])

        # Optimizer
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step = checkpoint['step']

        accelerator. print(f'Loaded at #{step}')

    # Train step
    def train_step():
        model.train()

        # Update LR
        if step < train_warmup_steps:
            lr = (lr_start + ((lr_max - lr_start) * step) / train_warmup_steps)
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            lr = lr / accelerator.num_processes
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0] / accelerator.num_processes

        # Load batch
        for _ in range(train_grad_accum_every):
            with accelerator.accumulate(model):
                with accelerator.autocast():

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
                        audio_full.append(a[:, :audio_split].to(device, non_blocking=True))
                        audio_partial.append(a[:, audio_split:].to(device, non_blocking=True))
                        audio_codecs.append(random.randint(1, 7))
                        texts.append(t.to(device, non_blocking=True))

                    # Forward
                    _, loss = model(
                        condition_text = texts,
                        condition_audio = audio_full,
                        audio = audio_partial,
                        codec = audio_codecs,
                        loss = True
                    )

                    # Backprop
                    optim.zero_grad()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_clip_grad_norm)
                    optim.step()

                    # Log skipping step
                    if optim.step_was_skipped:
                        accelerator.print("Step was skipped")

        return loss, lr

    #
    # Start Training
    #

    accelerator.print("Training started at step", step)
    while step < train_steps:
        
        # Step
        start = time.time()
        loss, lr = train_step()
        end = time.time()

        # Advance
        step = step + 1

        # Summary
        if step % train_log_every == 0:
            accelerator.log({
                "learning_rate": lr,
                "loss": loss
            }, step=step)
            accelerator.print(f'Step {step} | Loss: {loss} | LR: {lr}')

        # Save
        if step % train_save_every == 0:
            save()

    # End training
    if accelerator.is_main_process:
        accelerator.print("Finishing training...")
        save()
    accelerator.end_training()
    accelerator.print('✨ Training complete!')

#
# Utility
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    

if __name__ == "__main__":
    main()