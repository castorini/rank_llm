import contextlib
import logging
import os
import types

import datasets
import deepspeed
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import get_scheduler
from utils.data_utils import initialize_dataset_and_loader
from utils.model_utils import (
    NEFTune,
    initialize_model_and_tokenizer,
    initialize_optimizer,
    parse_args,
)
from utils.train_utils import (
    initialize_training_state,
    resume_from_checkpoint,
    save_checkpoint,
    train_epoch,
)

logger = get_logger(__name__)


def main():
    args = parse_args()

    # Initialize the accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        cpu=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    tokenizer, model = initialize_model_and_tokenizer(args)

    optimizer = initialize_optimizer(model, args.weight_decay, args.learning_rate)

    train_dataset, train_dataloader = initialize_dataset_and_loader(args, tokenizer)

    overrode_max_train_steps, num_update_steps_per_epoch = initialize_training_state(
        train_dataloader, args, accelerator
    )
    starting_epoch, resume_step, completed_steps = resume_from_checkpoint(
        args, num_update_steps_per_epoch, train_dataloader
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.noisy_embedding_alpha is not None:
        model = NEFTune(model, args.noisy_embedding_alpha)

    if args.with_tracking:
        experiment_config = vars(args)
        # Convert scheduler type to string for wandb
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("rank_llm", experiment_config)

        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    logger.info(
        "Starting accelerator.prepare() - this may take a while with DeepSpeed ZeRO-3"
    )
    train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, lr_scheduler
    )
    logger.info("Finished accelerator.prepare() successfully")

    # TODO: remove this when updated to newer accelerate.
    # This is a workaround, newer accelerate already addressed this https://github.com/huggingface/accelerate/issues/3481, just not pushed to PyPI yet.
    if (
        isinstance(model, deepspeed.DeepSpeedEngine)
        and model.zero_optimization_partition_gradients()
    ):

        def _null_no_sync(self):
            return contextlib.nullcontext()

        model.no_sync = types.MethodType(_null_no_sync, model)
        logger.info("Patched DeepSpeedEngine.no_sync → nullcontext for ZeRO-3")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.update(completed_steps)

    model.train()
    for epoch in range(starting_epoch, args.num_train_epochs):
        completed_steps = train_epoch(
            epoch=epoch,
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            tokenizer=tokenizer,
            args=args,
            device=accelerator.device,
            starting_epoch=starting_epoch,
            resume_step=resume_step,
            completed_steps=completed_steps,
            progress_bar=progress_bar,
            checkpointing_steps=args.checkpointing_steps,
        )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_checkpoint(model, tokenizer, accelerator, output_dir)

        if completed_steps >= args.max_train_steps:
            break

    if args.output_dir is not None:
        save_checkpoint(model, tokenizer, accelerator, args.output_dir)

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
