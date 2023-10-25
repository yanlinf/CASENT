import torch
import random
from transformers import get_scheduler
from transformers.optimization import Adafactor
import argparse
import numpy as np
import time
import logging
import os
import shutil
from accelerate import Accelerator
from accelerate.logging import get_logger
from casent.entity_typing_t5 import T5ForEntityTyping, T5ForEntityTypingPredictor, T5ForEntityTypingConfig, \
    get_dataloaders_t5
from casent.entity_typing_common import AutoTokenizerForEntityTyping
from casent.dataset import UFETDataset
from casent.utils.ufet_utils import evaluate_metrics
from casent.calibration import calibrate

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = get_logger(__name__)


def save_model_and_tokenizer(model, tokenizer, save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f'checkpoint saved to {save_dir}')


# `gradient_accumulation` is selected such that the model can be trained on a single RTX 2080 Ti
MODEL_DEFAULT_CONFIG = {
    't5-small': {'encoder_lr': 5e-5},
    't5-base': {'encoder_lr': 5e-5},
    't5-large': {'encoder_lr': 1e-5},
    't5-3b': {'encoder_lr': 1e-5, 'batch_size': 4, 'gradient_accumulation': 2, 'eval_batch_size': 4},
    'google/flan-t5-small': {'encoder_lr': 5e-5},
    'google/flan-t5-base': {'encoder_lr': 5e-5},
    'google/flan-t5-large': {'encoder_lr': 1e-5},
    'google/flan-t5-xl': {'encoder_lr': 1e-5, 'batch_size': 4, 'gradient_accumulation': 2, 'eval_batch_size': 4},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='t5-large', help='model name or path')
    parser.add_argument('-ds', '--dataset', default='ufet', choices=['ufet'])
    parser.add_argument('--ufet_dir', default='data/ufet/')
    parser.add_argument('--save_dir', default='checkpoints/entity_typing/')
    parser.add_argument('-optim', '--optim', default='adafactor')
    parser.add_argument('-elr', '--encoder_lr', default=1e-5, type=float)
    parser.add_argument('-sl', '--max_seq_length', default=128, type=int)
    parser.add_argument('-bs', '--batch_size', default=8, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=8, type=int)
    parser.add_argument('-ga', '--gradient_accumulation', default=1, type=int)
    parser.add_argument('--lr_schedule', default='constant',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                                 'constant_with_warmup', 'multi_step'])
    parser.add_argument('--lr_warmup_steps', default=0, type=int)
    parser.add_argument('--lr_ratio', default=0.1, type=float)
    parser.add_argument('--lr_epochs', default=[10], type=int, nargs='+')
    parser.add_argument('--eval_interval', type=int, default=None)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--patience', default=5, type=int, help='stop if dev f1 does not increase for N epochs')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-c', '--calibrate', default='prior_platt',
                        choices=['disabled', 'single', 'prior_platt'])
    parser.add_argument('--decode_prefix', default='In this sentence, {} is')
    parser.add_argument('--n_decode', type=int, default=24)
    parser.add_argument('--n_beams', type=int, default=24)

    parser.add_argument('--train_with_dev', action='store_true')
    parser.add_argument('--calibrate_with_constrained_beam_search', action='store_true')
    parser.add_argument('--subsample_dev', default=-1.0, type=float)
    args = parser.parse_args()
    parser.set_defaults(**MODEL_DEFAULT_CONFIG.get(args.model, {}))

    args = parser.parse_args()
    logging.basicConfig(
        format='%(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    train(args)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    accelerator = Accelerator(split_batches=True, gradient_accumulation_steps=args.gradient_accumulation)

    logger.info(args)
    logger.info('')
    accelerator.wait_for_everyone()
    logger.info(accelerator.state, main_process_only=False)

    tokenizer = AutoTokenizerForEntityTyping.from_pretrained(args.model)

    train_dataloader, dev_dataloader, test_dataloader = get_dataloaders_t5(
        args.dataset, tokenizer, args.decode_prefix,
        args.ufet_dir,
        args.max_seq_length, args.batch_size,
        args.eval_batch_size,
        train_with_dev=args.train_with_dev,
        subsample_dev=args.subsample_dev,
    )

    # for calibration and evaluation
    type_vocab = UFETDataset.get_type_vocab(args.ufet_dir)
    type_freq = UFETDataset.get_type_freq(args.ufet_dir)

    config_kwargs = {
        'pred_num_decode': args.n_decode,
        'pred_num_beams': args.n_beams,
        'entity_max_length': args.max_seq_length,
        'decode_prefix': args.decode_prefix,
        'calibration': args.calibrate,
        'type_vocab': type_vocab,
        'oov_type': 'businesswoman',
    }
    config = T5ForEntityTypingConfig.from_pretrained(args.model, **config_kwargs)
    model = T5ForEntityTyping.from_pretrained(args.model, config=config)

    if not os.path.isdir(args.model):
        model.resize_token_embeddings(len(tokenizer))

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    all_params = [(n, p) for n, p in model.named_parameters()]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in all_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in all_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.encoder_lr}
    ]

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    elif args.optim == 'adafactor':
        optimizer = Adafactor(optimizer_grouped_parameters, relative_step=False, scale_parameter=False)
    else:
        raise ValueError(f'Unknown optimizer class {args.optim}')

    # Scheduler and math around the number of training steps.
    if args.lr_schedule == 'multi_step':
        n_steps_per_epoch = (len(train_dataloader) + args.gradient_accumulation - 1) // args.gradient_accumulation
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[n_steps_per_epoch * e for e in args.lr_epochs],
            gamma=args.lr_ratio
        )
    else:
        lr_scheduler = get_scheduler(name=args.lr_schedule, optimizer=optimizer,
                                     num_warmup_steps=args.lr_warmup_steps,
                                     num_training_steps=args.n_epochs * len(train_dataloader))

    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader)

    if accelerator.is_main_process:
        if os.path.isdir(args.save_dir) and len(os.listdir(args.save_dir)) > 0:
            shutil.rmtree(args.save_dir)

    # Train!
    logger.info('')
    logger.info('***** Running training *****')
    logger.info(f'  Num train examples = {len(train_dataloader.dataset)}')
    logger.info(f'  Num dev examples = {len(dev_dataloader.dataset)}')
    logger.info(f'  Num test examples = {len(test_dataloader.dataset)}')
    logger.info(f'  Num Epochs = {args.n_epochs}')
    logger.info(f'  Batch size = {args.batch_size}x{args.gradient_accumulation}')
    logger.info('****************************')

    best_dev_result = None
    best_test_result = None
    best_epoch = -1
    step = 0
    training_start_time = time.time()
    t0, step_since_last_eval, total_loss = time.time(), 0, 0
    for epoch in range(args.n_epochs):
        if epoch - best_epoch > args.patience:
            break
        model.train()
        n_iter = 0

        for batch in train_dataloader:
            n_iter += 1

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if n_iter % args.gradient_accumulation == 0 or n_iter == len(train_dataloader):
                step_since_last_eval += 1
                step += 1

            if step_since_last_eval > 0 and (n_iter == len(train_dataloader) or (
                    args.eval_interval is not None and step % args.eval_interval == 0)):
                ms_per_batch = (time.time() - t0) / step_since_last_eval * 1000
                dev_result = None

                predictor = T5ForEntityTypingPredictor(
                    accelerator.unwrap_model(model), tokenizer, config
                )
                if args.calibrate is not None:
                    dev_result = calibrate(accelerator, predictor, dev_dataloader, type_vocab,
                                           type_freq, args.calibrate, tokenizer,
                                           with_constrained_beam_search=args.calibrate_with_constrained_beam_search)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if dev_result is None:
                        dev_result = evaluate_metrics(predictor, dev_dataloader)
                    test_result = evaluate_metrics(predictor, test_dataloader)
                    total_loss /= step_since_last_eval
                    elr = lr_scheduler.get_last_lr()[0]
                    logger.info(f'epoch: {epoch}  step: {step}  loss: {total_loss:.2e}  dev_f1: {dev_result["f"]:.4f}'
                                f'  test_f1: {test_result["f"]:.4f}  test_p: {test_result["p"]:.4f}  test_r: {test_result["r"]:.4f}'
                                f'  elr: {elr:.2e}  ms/batch: {ms_per_batch:.1f}')
                    t0, step_since_last_eval, total_loss = time.time(), 0, 0

                    if best_dev_result is None or dev_result['f'] > best_dev_result['f']:
                        best_dev_result = dev_result
                        best_test_result = test_result
                        best_epoch = epoch
                        save_model_and_tokenizer(accelerator.unwrap_model(model), tokenizer, args.save_dir)
                accelerator.wait_for_everyone()
                model.train()

    if accelerator.is_main_process:
        logger.info(f'***** training ends *****')
        logger.info('')
        logger.info('training time: {:.2f} seconds'.format(time.time() - training_start_time))
        logger.info('best epoch: {}'.format(best_epoch))
        logger.info('best dev p: {:.4f}'.format(best_dev_result['p']))
        logger.info('best dev r: {:.4f}'.format(best_dev_result['r']))
        logger.info('best dev f1: {:.4f}'.format(best_dev_result['f']))
        logger.info('best test p: {:.4f}'.format(best_test_result['p']))
        logger.info('best test r: {:.4f}'.format(best_test_result['r']))
        logger.info('best test f1: {:.4f}'.format(best_test_result['f']))
        logger.info('')


if __name__ == '__main__':
    main()
