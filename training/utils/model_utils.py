import argparse

import bitsandbytes as bnb
import torch
from accelerate.logging import get_logger
from torch import nn
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
)
from transformers.trainer_pt_utils import get_parameter_names

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        required=True,
        help="Training dataset path in jsonl format",
    )
    parser.add_argument("--cache_dir", type=str, help="Path to cache")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--noisy_embedding_alpha",
        type=int,
        default=None,
        help="NEFT https://arxiv.org/abs/2310.05914, set this to a number (paper default is 5) to add noise to embeddings",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--ranking_loss",
        type=str,
        default="lambda",
        help="Ranking loss to use",
        choices=["lambda", "listnet", "ranknet"],
    )
    parser.add_argument(
        "--weighted", action="store_true", help="Use weighting with Ranknet"
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="generation",
        help="Training objective for reranker training",
        choices=["ranking", "generation", "combined"],
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=True,
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Option to create the model as an empty shell for lower RAM consumption.",
    )
    args = parser.parse_args()
    assert args.output_dir is not None
    return args


def NEFTune(model, noise_alpha=5):
    """
    Apply noisy embeddings during training (NEFTune technique).
    """

    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha / torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(
                    -mag_norm, mag_norm
                )
            else:
                return orig_embed(x)

        return new_func

    orig_forward = model.base_model.embed_tokens.forward
    model.base_model.embed_tokens.forward = noised_embed(orig_forward, noise_alpha)
    return model


def initialize_model_and_tokenizer(args):
    """
    Initialize the model, tokenizer, and config based on provided arguments.
    """
    logger.info("Starting model and tokenizer initialization...")
    if args.resume_from_checkpoint:
        config = AutoConfig.from_pretrained(
            args.resume_from_checkpoint,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.resume_from_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(
            args.resume_from_checkpoint,
            use_fast=True,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=True,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.resume_from_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(
            args.resume_from_checkpoint,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        logger.info(f"Loading model from {args.model_name_or_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
        logger.info("Model loaded successfully!")
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(
            config,
            cache_dir=args.cache_dir,
            attn_implementation="flash_attention_2",
            trust_remote_code=args.trust_remote_code,
        )

    # Handle pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})

    # Resize embeddings if necessary
    logger.info("Checking tokenizer and model compatibility...")
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(f"Resizing token embeddings from {embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    logger.info("Model and tokenizer initialization completed successfully!")
    return tokenizer, model


def initialize_optimizer(model, weight_decay, learning_rate):
    """
    Initialize the optimizer with the appropriate parameter groups.
    """
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=learning_rate,
    )

    return optimizer
