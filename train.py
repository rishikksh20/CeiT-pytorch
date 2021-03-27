# coding=utf-8


from __future__ import absolute_import, division, print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import HParam
from ceit import CeiT
from utils import WarmupLinearSchedule, WarmupCosineSchedule
from data_utils import get_loader
from utils import get_world_size, get_rank


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(name, outdir, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(outdir, "%s_checkpoint.pyt" % name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", outdir)





def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(hp):
    random.seed(hp.train.seed)
    np.random.seed(hp.train.seed)
    torch.manual_seed(hp.train.seed)
    if hp.train.ngpu > 0:
        torch.cuda.manual_seed_all(hp.train.seed)


def valid(device, local_rank, hp, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", hp.train.valid_batch)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(local_rank, args, hp, model):
    
    if hp.train.ngpu > 1:
        dist.init_process_group(backend="nccl", init_method="tcp://localhost:54321",
                           world_size=hp.train.ngpu, rank=local_rank)
        
    torch.cuda.manual_seed(hp.train.seed)
    device = torch.device('cuda:{:d}'.format(local_rank))
    model = model.to(device)
    
    
    """ Train the model """
    if local_rank in [-1, 0]:
        os.makedirs(hp.data.outdir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        print("Loading dataset :")

    hp.train.batch = hp.train.batch // hp.train.accum_grad

    # Prepare dataset
    train_loader, test_loader = get_loader(local_rank, hp)

    # Prepare optimizer and scheduler
    
    optimizer = torch.optim.AdamW(model.parameters(), hp.train.lr, betas=[0.8, 0.99], weight_decay=0.05)
    t_total = hp.train.num_steps
    if hp.train.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=hp.train.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=hp.train.warmup_steps, t_total=t_total)

    

    # Distributed training
    if hp.train.ngpu > 1:
        model = DDP(model, device_ids=[local_rank])

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", hp.train.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", hp.train.batch)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                hp.train.batch * hp.train.accum_grad * (
                    hp.train.ngpu if local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", hp.train.accum_grad)

    model.zero_grad()
    set_seed(hp)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    
    loss_fct = torch.nn.CrossEntropyLoss()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            logits = model(x)
            
            loss = loss_fct(logits.view(-1, hp.model.num_classes), y.view(-1))

            if hp.train.accum_grad > 1:
                loss = loss / hp.train.accum_grad
            loss.backward()

            if (step + 1) % hp.train.accum_grad == 0:
                losses.update(loss.item()*hp.train.accum_grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.train.grad_clip)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % hp.train.valid_step == 0 and local_rank in [-1, 0]:
                    accuracy = valid(device, local_rank, hp, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args.name, hp.data.outdir, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.") 
    args = parser.parse_args()
    
    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    # Setup CUDA, GPU & distributed training
    if hp.train.ngpu > 1:
        torch.cuda.manual_seed(hp.train.seed)
        hp.train.ngpu = torch.cuda.device_count()
        hp.train.batch = int(hp.train.batch / hp.train.ngpu)
        print('Batch size per GPU :', hp.train.batch)
    

    # Set seed
    set_seed(hp)

    # Model & Tokenizer Setup
    model = CeiT(image_size = hp.data.image_size, patch_size = hp.model.patch_size, num_classes = hp.model.num_classes, 
                 dim = hp.model.dim, depth = hp.model.depth, heads = hp.model.heads, pool = hp.model.pool, 
                 in_channels = hp.model.in_channels, out_channels = hp.model.out_channels, with_lca=hp.model.with_lca)
    
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)

    # Training
    #train(args, hp, model)
    if hp.train.ngpu > 1:
        mp.spawn(train, nprocs=hp.train.ngpu, args=(args, hp, model,))
    else:
        train(0, args, hp, model)


if __name__ == "__main__":
    main()
