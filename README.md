# DDP
This is the repo for training large language models with DDP (distributed data parallel). Training becomes infeasible on a single gpu when the model size is large. DDP is a way to train large models on multiple gpus.

## Usage
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the `setup.py` script
```bash
python3 setup.py
```

## Data
Currently, the repo supports `huggingface datasets`. You can use any dataset from the [datasets hub](https://huggingface.co/datasets). For example, if you want to use the `wikitext-2` dataset, you have to make the following changes:
1. In the `setup.py` file, set the variable `DATASET_NAME = wikitext-2`.
2. Run the setup script as instructed above.