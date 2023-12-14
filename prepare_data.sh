red_pyjama="togethercomputer/RedPajama-Data-1T-Sample"
tiny_stories="roneneldan/TinyStories"
num_proc=48
num_partitions=48
max_seq_len=1024
tokenizer="meta-llama/Llama-2-7b-hf"
dataset_text_field="text"
subset=0.0
save=1 # 0 for false, 1 for true
save_dir="/dumps"

cd utils

python3 data.py --dataset_name $red_pyjama \
	--subset $subset \
	--dataset_text_field $dataset_text_field \
	--tokenizer_name $tokenizer \
	--max_length $max_seq_len \
	--num_proc $num_proc \
	--num_partitions $num_partitions \
	--split_rows \
	--save $save \
	--save_dir $save_dir