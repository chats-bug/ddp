#!/bin/bash

fsdp=true
nproc_per_node=2
seq_len=1024
global_batch_size=24
bsz=6
gas=4
lr=0.0003
max_grad_norm=1.0
local_path=/dumps/TinyStories_all_prepared.pt
dtype=fp16
eval_steps=1000
save_steps=20000
report_to=null
wandb_project=fsdp-trainer

# if global batch size is not 0, the override the gas as gas=global_batch_size / bsz
if [ $global_batch_size -ne 0 ]; then
    gas=$(($global_batch_size / $bsz))
fi

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
        --eval_every $eval_steps \
        --save_every $save_steps \
        --report_to $report_to \
        --wandb_project $wandb_project \
		--fsdp
fi

python3 setup.py \
    --seq_len $seq_len \
    --batch_size $bsz \
    --grad_accumulation_steps $gas \
    --num_epochs 1 \
    --no-small_model \
    --lr $lr \
    --max_grad_norm $max_grad_norm \
    --local_path $local_path \
    --torch_dtype $dtype \
    --eval_every $eval_steps \
    --save_every $save_steps \
    --report_to $report_to \
    --wandb_project $wandb_project \