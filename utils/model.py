from typing import Optional

from rich.console import Console
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig

from .utils import num_trainable_params

console = Console()
LLAMA_TOKENIZER_PATH = "meta-llama/Llama-2-7b-hf"


def get_llama_model(
    config: Optional[LlamaConfig] = None, special_tokens: Optional[dict] = None
):
    assert isinstance(config, LlamaConfig), "Only LlamaConfig is supported for now"

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_PATH)
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    if config is None:
        # warn the user that the default config is used
        console.log("No config is provided, using the default config", style="bold red")
        config = LlamaConfig(
            vocab_size=tokenizer.voacab_size,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
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
        model = AutoModelForCausalLM.from_config(config)
        console.log(f"Default config: {config}", style="bold red")
        console.log(f"Model Size: {num_trainable_params(model)}", style="bold red")
    else:
        model = AutoModelForCausalLM.from_config(config)
    return {"model": model, "tokenizer": tokenizer}
