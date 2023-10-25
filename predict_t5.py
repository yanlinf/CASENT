import torch
import random
import argparse
import numpy as np
import os
from tqdm import tqdm
import json
import logging
from casent.entity_typing_t5 import T5ForEntityTyping, T5ForEntityTypingPredictor, T5ForEntityTypingConfig, \
    get_dataloaders_t5
from casent.dataset import UFETDataset
from casent.entity_typing_common import AutoTokenizerForEntityTyping
from casent.utils.ufet_utils import evaluate_metrics, evaluate_group_metrics

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/exp3_2/')
    parser.add_argument('-ds', '--dataset', default='ufet', choices=['ufet'])
    parser.add_argument('--ufet_dir', default='data/ufet/')
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # parser.add_argument('-c', '--calibrate', default=None, choices=['disabled', 'group', 'single', 'prior_platt'])

    # # arguments valid only for t5
    # parser.add_argument('--n_decode', type=int, default=24)
    # parser.add_argument('--n_beams', type=int, default=24)
    args = parser.parse_args()

    logging.basicConfig(
        format='%(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    logger.info(args)
    logger.info('')
    predict(args)


def run_inference(predictor, dataloader, predict_kwargs={}, output_path=None, progress_bar=False):
    outputs = []
    for batch in tqdm(dataloader, disable=not progress_bar):
        for i, pred in enumerate(predictor.predict_batch(
                **batch, **predict_kwargs,
                # do_calibration=False, do_thresholding=False
        )):
            types = pred.types
            scores = pred.scores
            # scores = [np.exp(s) for s in scores]
            # type2score = {t: s for t, s in zip(types, scores) if s > 0.24603667}
            type2score = {t: s for t, s in zip(types, scores)}
            types = sorted(type2score, key=type2score.get, reverse=True)
            outputs.append({
                'id': batch['id'][i],
                'context': batch['context'][i],
                'predictions': {t: type2score[t] for t in types},
                'labels': batch['raw_labels'][i],
            })
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=2)


def predict(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizerForEntityTyping.from_pretrained(args.model)

    predictor = T5ForEntityTypingPredictor.from_pretrained(args.model)
    predictor.to(torch.device('cuda:0'))

    dev_dataloader, test_dataloader = get_dataloaders_t5(
        args.dataset, tokenizer, predictor.config.decode_prefix,
        args.ufet_dir,
        max_length=args.max_seq_length,
        eval_batch_size=args.eval_batch_size,
        splits=('dev', 'test')
    )
    for split, dataloader in [('dev', dev_dataloader), ('test', test_dataloader)]:
        # output_path = os.path.join(args.model, f'{split}_pred_no_cali.json')
        output_path = os.path.join(args.model, f'{split}_pred.json')
        run_inference(predictor, dataloader, output_path=output_path, progress_bar=True)
        print('Saved predictions to', output_path)


if __name__ == '__main__':
    main()
