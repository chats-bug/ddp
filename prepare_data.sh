cd utils

python3 data.py --dataset_name "roneneldan/TinyStories" \
	--subset 0.0 \
	--dataset_text_field "text" \
	--tokenizer_name "meta-llama/Llama-2-7b-hf" \
	--max_length 1024 \
	--num_proc 64 \
	--num_partitions 64 \
	--save 1 \
	--save_dir "/dumps"