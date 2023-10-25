import json
import numpy as np
import os
from typing import List
import argparse
from casent.calibration import *


def convert_to_array(jsonl_path: str, type_vocab: List[str]) -> np.ndarray:
    # compute the number of lines in the file
    with open(jsonl_path, 'r') as fin:
        n_lines = sum(1 for _ in fin)
    all_scores = np.zeros((n_lines, len(type_vocab)), dtype=float)
    type2idx = {t: i for i, t in enumerate(type_vocab)}
    with open(jsonl_path, 'r') as fin:
        for i, line in tqdm(enumerate(fin), total=n_lines):
            dic = json.loads(line)
            type2scores = dic['confidence_ranking']
            for t, s in type2scores.items():
                all_scores[i, type2idx[t]] = s
    return all_scores


def get_all_labels(split, ufet_dir, type_vocab) -> np.ndarray:
    ds = UFETDataset(split, ufet_dir)
    all_labels = np.zeros((len(ds), len(type_vocab)), dtype=float)
    type2idx = {t: i for i, t in enumerate(type_vocab)}
    for i, example in enumerate(ds):
        for t in example.labels:
            all_labels[i, type2idx[t]] = 1
    return all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ufet_dir', default='data/ufet/')
    parser.add_argument('--dev',
                        default='data/lite_output/epochs1650_batch16_margin0.1_lr5e-06_15_27_30_Sep_09_2021dev_processed.json')
    parser.add_argument('--test',
                        default='data/lite_output/epochs1650_batch16_margin0.1_lr5e-06_15_27_30_Sep_09_2021_test_processed.json')
    parser.add_argument('--n_bins', default=10, type=int)
    args = parser.parse_args()

    type_vocab = UFETDataset.get_type_vocab(args.ufet_dir)

    dev_all_scores = convert_to_array(args.dev, type_vocab)
    test_all_scores = convert_to_array(args.test, type_vocab)

    dev_all_labels = get_all_labels('dev', args.ufet_dir, type_vocab)
    test_all_labels = get_all_labels('test', args.ufet_dir, type_vocab)

    threshold, metrics = grid_search_group_threshold(
        dev_all_scores, dev_all_labels,
        group_idx=[np.arange(len(type_vocab))],
        n_grids=200
    )
    test_pred = (test_all_scores > threshold).astype(float)
    p, r, f1 = ufet_f1_array(test_pred, test_all_labels)
    print()
    print(f'Threshold: {threshold}')  #  0.93220661
    print()
    print('dev p: {:.4f}'.format(metrics['p']))
    print('dev r: {:.4f}'.format(metrics['r']))
    print('dev f1: {:.4f}'.format(metrics['f']))
    print('test p: {:.4f}'.format(p))
    print('test r: {:.4f}'.format(r))
    print('test f1: {:.4f}'.format(f1))

    output_dir = os.path.dirname(args.dev)

    for split, all_scores, all_labels in zip(
            ('dev', 'test'),
            (dev_all_scores, test_all_scores),
            (dev_all_labels, test_all_labels)
    ):
        output_path = os.path.join(output_dir, f'{split}_calibration_curve.png')
        plot_ufet_group_calibration_curves(all_scores, all_labels,
                                output_path=output_path,
                                n_bins=5)
        print()
        print(f'Calibration curve saved to {output_path}')

        metrics = evaluate_calibration_error(
            all_scores, all_labels, n_bins=args.n_bins, prediction_mask=(all_scores > threshold)
        )
        print()
        print(f'Calibration results on {split} set:')
        print('Expected Calibration Error (ECE): {:.4f}'.format(metrics['ece']))
        print('Maximum Calibration Error (MCE): {:.4f}'.format(metrics['mce']))
        print('Total Calibration Error (TCE): {:.4f}'.format(metrics['tce']))
        print()


if __name__ == "__main__":
    main()
