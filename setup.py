import torch
import torch.optim
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from rich.console import Console

from datautils import get_dataset, CustomDataset, PoorMansDataLoader
from single_gpu import Trainer

console = Console()

SEQ_LEN = 512
BATCH_SIZE = 8
LR = 3e-4
DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"


def load_train_objs():
    console.log("Loading dataset...")
    hf_dataset = get_dataset(DATASET_NAME, split="train")
    console.log("Splitting dataset...")
    hf_dataset = hf_dataset.train_test_split(test_size=0.001)
    train_dataset = hf_dataset["train"]
    val_dataset = hf_dataset["test"]

    console.log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    console.log("Making custom dataset...")
    train_dataset = CustomDataset(
        train_dataset, tokenizer, seq_len=SEQ_LEN, dataset_text_field="text"
    )
    val_dataset = CustomDataset(
        val_dataset, tokenizer, seq_len=SEQ_LEN, dataset_text_field="text"
    )

    console.log("Loading model...")
    larger_model = AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=32,
            hidden_act="silu",
            max_position_embeddings=SEQ_LEN,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
        )
    )
    smaller_model = AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=512,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
        )
    )
    model = smaller_model
    console.log("Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_mult=1, T_0=50, eta_min=0.1 * LR, last_epoch=-1
    )

    return train_dataset, val_dataset, model, optimizer, lr_scheduler


def prepare_dataloader(dataset: CustomDataset, batch_size: int):
    return PoorMansDataLoader(dataset, batch_size=batch_size)


def num_trainable_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    train_dataset, val_dataset, model, optimizer, lr_scheduler = load_train_objs()
    train_dataloader = prepare_dataloader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = prepare_dataloader(val_dataset, batch_size=BATCH_SIZE)

    console.log(f"Model Specs: {model.config}")
    console.log(f"Trainable Parameters: {num_trainable_params(model):,}")

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_schedular=lr_scheduler,
        gpu_id=0,
        device=device,
        eval_every=10000,
        save_every=10000,
        log_every=1,
        grad_accumulation_steps=4,
        torch_dtype=torch.float16,
    )

    console.log(f"Starting training with {device}...")
    trainer.train(max_epochs=1)
