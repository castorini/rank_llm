import os
import torch
from torch import nn
import math
from tqdm import tqdm
from accelerate.logging import get_logger

from .loss import lambdarank, listNet, rank_net

logger = get_logger(__name__)
START_IDX = ord('A')

def compute_generation_loss(model_outputs, labels, device):
    """Compute the loss for generation/language modeling task."""
    logits = model_outputs.logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    log_probs = -nn.functional.log_softmax(logits, dim=-1)
    
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    epsilon = 0.0
    ignore_index = -100
    padding_mask = labels.eq(ignore_index)

    # Handle -100 indices
    labels = torch.clamp(labels, min=0)
    nll_loss = log_probs.gather(dim=-1, index=labels)
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Calculate mean loss over active elements
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
    
    loss = (1 - epsilon) * nll_loss + epsilon * smoothed_loss
    return loss, loss  # Return both total loss and generation loss

def compute_ranking_loss(model_outputs, rank_labels, source_lens, tokenizer, args, device):
    """Compute the loss for ranking task."""
    rank_losses = torch.zeros(len(rank_labels))
    
    for batch_index, true_rank in enumerate(rank_labels):
        pred_logits = model_outputs.logits[batch_index]
        rank_pred_index = source_lens[batch_index]
        pred = pred_logits[rank_pred_index]
        
        gather_indices = [tokenizer.convert_tokens_to_ids(chr(c)) 
                         for c in range(START_IDX, START_IDX + len(true_rank))]
        scores = torch.gather(pred, 0, torch.tensor(gather_indices).to(device))
        scores_sorted, _ = scores.sort(descending=True, dim=-1)
        true_scores = torch.gather(scores_sorted, 0, torch.tensor(true_rank).to(device))
        
        if args.ranking_loss == "lambda":
            rank_losses[batch_index] = lambdarank(scores.unsqueeze(0), true_scores.unsqueeze(0))
        elif args.ranking_loss == "listnet":
            rank_losses[batch_index] = listNet(scores.unsqueeze(0), true_scores.unsqueeze(0))
        elif args.ranking_loss == "ranknet":
            rank_losses[batch_index] = rank_net(scores.unsqueeze(0), true_scores.unsqueeze(0), 
                                              weighted=args.weighted)
    
    rank_loss = rank_losses.mean().to(device)
    return rank_loss

def process_batch(batch, model, tokenizer, args, device):
    """Process a single batch and compute the appropriate loss."""
    if args.objective == "generation" or args.objective == "combined":
        if args.objective == "generation":
            tokenized_input, label = batch
            tokenized_input = tokenized_input.to(device)
            outputs = model(tokenized_input)
            loss, generate_loss = compute_generation_loss(outputs, label, device)
            rank_loss = None
        else:  # combined
            tokenized_input, label, rank_labels, source_lens = batch
            tokenized_input = tokenized_input.to(device)
            outputs = model(tokenized_input)
            loss, generate_loss = compute_generation_loss(outputs, label, device)
            rank_loss = compute_ranking_loss(outputs, rank_labels, source_lens, tokenizer, args, device)
            
            if not torch.isnan(rank_loss):
                if args.ranking_loss == "listnet":
                    loss += torch.mul(rank_loss, 0.1)
                elif args.ranking_loss == "ranknet" and args.weighted:
                    loss += torch.mul(rank_loss, 10.0)
                else:
                    loss += rank_loss
    else:  # pure ranking
        tokenized_input, label = batch
        tokenized_input = tokenized_input.to(device)
        outputs = model(**tokenized_input)
        loss = compute_ranking_loss(outputs, [label], [0], tokenizer, args, device)
        generate_loss = None
        rank_loss = loss
        
    return loss, generate_loss, rank_loss

def save_checkpoint(model, tokenizer, accelerator, output_dir):
    """Save model checkpoint."""
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model)
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

def train_epoch(epoch, model, train_dataloader, optimizer, lr_scheduler, accelerator, 
                tokenizer, args, device, starting_epoch, resume_step, completed_steps,
                progress_bar, checkpointing_steps):
    """Execute one training epoch."""
    if args.with_tracking:
        step_loss = 0
        if args.objective == "combined":
            step_rank_loss = 0
            step_generate_loss = 0
            
    if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
        active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
    else:
        active_dataloader = train_dataloader

    for step, batch in enumerate(active_dataloader):
        with accelerator.autocast():
            with accelerator.accumulate(model):
                loss, generate_loss, rank_loss = process_batch(batch, model, tokenizer, args, device)
                
                if args.with_tracking:
                    step_loss += loss.detach().float()
                    if args.objective == "combined":
                        if rank_loss is not None:
                            step_rank_loss += rank_loss.detach().float()
                        if generate_loss is not None:
                            step_generate_loss += generate_loss.detach().float()
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        if accelerator.sync_gradients:
            completed_steps += 1
            if args.with_tracking:
                accelerator.log({"train_loss": step_loss}, step=completed_steps)
                if args.objective == "combined":
                    accelerator.log({
                        "generate_loss": step_generate_loss,
                        "rank_loss": step_rank_loss,
                    }, step=completed_steps)
                    step_rank_loss = 0
                    step_generate_loss = 0
                step_loss = 0
            progress_bar.update(1)

        if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
            output_dir = f"step_{completed_steps}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_checkpoint(model, tokenizer, accelerator, output_dir)

        if completed_steps >= args.max_train_steps:
            break
            
    return completed_steps

def initialize_training_state(train_dataloader, args):
    """Initialize training state variables."""
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    return overrode_max_train_steps, num_update_steps_per_epoch

def resume_from_checkpoint(args, num_update_steps_per_epoch, train_dataloader):
    """Handle checkpoint resumption logic."""
    if not args.resume_from_checkpoint:
        return 0, None, 0
        
    if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "":
        checkpoint_path = args.resume_from_checkpoint
    else:
        # Get the most recent checkpoint
        dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        checkpoint_path = dirs[-1]
        
    path = os.path.basename(checkpoint_path)
    logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    
    training_difference = os.path.splitext(path)[0]
    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch
    else:
        resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // args.gradient_accumulation_steps
        resume_step -= starting_epoch * len(train_dataloader)
        
    return starting_epoch, resume_step, completed_steps 