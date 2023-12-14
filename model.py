from transformers import LlamaConfig

from utils import get_llama_model


def llama_1_7_model(seq_len: int = 1024, special_tokens: dict = None):
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=seq_len,
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
    return get_llama_model(config, special_tokens)


def smaller_llama(seq_len: int = 512, special_tokens: dict = None):
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=seq_len,
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
    return get_llama_model(config, special_tokens)


def original_llama(seq_len: int = 1024, special_tokens: dict = None):
    config = None
    return get_llama_model(config, special_tokens)
