### first run the data prep
# cd utils
# python3 data.py python3 utils/data.py --save 1 --num_proc [num_cores] --num_partitions [num_partitions] --seq_len [seq_len]

torchrun --standalone --nproc_per_node=2 setup.py --seq_len 512 --batch_size 2 --grad_accumulation_steps 8 --num_epochs 1 --no-small_model --lr 0.0003 --max_grad_norm 1.0 --local_path --torch_dtype fp32 --fsdp