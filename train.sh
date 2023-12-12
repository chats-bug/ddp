#!/bin/bash

fsdp=true
nproc_per_node=4
seq_len=1024
bsz=3
gas=16
lr=0.0003
max_grad_norm=1.0
local_path=/root/ddp/dumps/TinyStories_all_prepared.pt
dtype=bf16
wandb_project=fsdp-trainer
report_to=wandb

## num steps should be 10,360: 1 epoch; batch = 2 * 24

if $fsdp; then
    torchrun --standalone --nproc_per_node=$nproc_per_node setup.py \
        --seq_len $seq_len \
        --batch_size $bsz \
        --grad_accumulation_steps $gas \
        --num_epochs 1 \
        --no-small_model \
        --lr $lr \
        --max_grad_norm $max_grad_norm \
        --local_path $local_path \
        --torch_dtype $dtype \
        --report_to $report_to \
        --wandb_project $wandb_project \
		--fsdp
fi

python3 setup.py \
    --seq_len $seq_len \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --num_epochs 1 \
    --no-small_model \
    --lr 0.0003 \
    --max_grad_norm 1.0 \
    --local_path $local_path \
    --torch_dtype $dtype