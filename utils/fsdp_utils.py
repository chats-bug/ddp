import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
import functools
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)
from rich.console import Console


console = Console()


def prepare_model_for_fsdp(model, **kwargs):
	"""
	Prepare a model for FSDP training.
	"""
	wrap_policy = kwargs.pop("wrap_policy", None)
	transformer_layer_class = kwargs.pop("transformer_layer_cls", None)
	wrap_policy_args = kwargs.pop("wrap_policy_args", {})

	assert transformer_layer_class is not None, "transformer_layer_cls must be specified"

	if wrap_policy:
		fsdp_wrap_policy = functools.partial(
			wrap_policy,
			transformer_layer_cls={
				transformer_layer_class,
			},
			**wrap_policy_args,
		)
	else:
		fsdp_wrap_policy = functools.partial(
			transformer_auto_wrap_policy,
			transformer_layer_cls={
				transformer_layer_class,
			}
		)
	
	torch_dtype = kwargs.pop("torch_dtype", None)
	if not torch_dtype:
		console.log("torch_dtype not specified, defaulting to torch.float32")
		torch_dtype = torch.float32
	
	mixed_precision_args = kwargs.pop("mixed_precision_args", {})
	mixed_precision = MixedPrecision(
		param_dtype=torch_dtype,
		reduce_dtype=torch_dtype,
		buffer_dtype=torch_dtype,
		**mixed_precision_args,
	)

	backward_prefetch = kwargs.pop("backward_prefetch", None)
	if backward_prefetch == "pre":
		backward_prefetch = BackwardPrefetch.BACKWARD_PRE
	elif backward_prefetch == "post":
		backward_prefetch = BackwardPrefetch.BACKWARD_POST
	else:
		backward_prefetch = BackwardPrefetch.OFF
	
	sharding_strategy = kwargs.pop("sharding_strategy", None)
	if sharding_strategy == "full":
		sharding_strategy = ShardingStrategy.FULL
	elif sharding_strategy == "grad_op":
		sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
	elif sharding_strategy == "hybrid":
		sharding_strategy = ShardingStrategy.HYBRID_SHARD
	elif sharding_strategy == "hybrid_zero2":
		sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
	else:
		sharding_strategy = ShardingStrategy.OFF

	model = FSDP(
		module=model,
		mixed_precision=mixed_precision,
		auto_wrap_policy=fsdp_wrap_policy,
		backward_prefetch=backward_prefetch,
		sharding_strategy=sharding_strategy,
		**kwargs,
	)

	console.print("FSDP Model Args:")
	console.print(f"mixed_precision: {mixed_precision}")
	console.print(f"auto_wrap_policy: {fsdp_wrap_policy}")
	console.print(f"transformer_layer_class: {transformer_layer_class}")
	console.print(f"backward_prefetch: {backward_prefetch}")

	return model