import argparse
import json
import numpy as np
import logging
from casent.calibration import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default='checkpoints/exp3_2/test_pred.json')
    parser.add_argument('--ufet_dir', default='data/ufet/')
    parser.add_argument('--n_bins', default=10, type=int)
    args = parser.parse_args()

    print(args)

    # --------------------------------------
    # Load model predictions
    # --------------------------------------

    type_vocab = UFETDataset.get_type_vocab(args.ufet_dir)

    type2idx = {t: i for i, t in enumerate(type_vocab)}
    with open(args.input_path, 'r') as fin:
        pred = json.load(fin)

        all_labels = np.zeros((len(pred), len(type_vocab)), dtype=float)
        all_scores = np.full((len(pred), len(type_vocab)), -np.inf, dtype=float)

        for i, dic in enumerate(pred):
            type2scores = dic['predictions']
            for t, s in type2scores.items():
                all_scores[i, type2idx[t]] = s
            for t in dic['labels']:
                all_labels[i, type2idx[t]] = 1.

    # --------------------------------------
    # Evaluate task performance (p, r, f1)
    # --------------------------------------

    all_pred = (all_scores > -np.inf).astype(float)
    p, r, f1 = ufet_f1_array(all_pred, all_labels)
    print()
    print('p: {:.4f}'.format(p))
    print('r: {:.4f}'.format(r))
    print('f1: {:.4f}'.format(f1))

    # --------------------------------------
    # Evaluate calibration error
    # --------------------------------------

    prediction_mask = all_scores > -np.inf

    metrics = evaluate_calibration_error(all_scores, all_labels,
                                         n_bins=args.n_bins, prediction_mask=prediction_mask)
    print()
    print('Expected Calibration Error (ECE): {:.4f}'.format(metrics['ece']))
    print('Maximum Calibration Error (MCE): {:.4f}'.format(metrics['mce']))
    print('Total Calibration Error (TCE): {:.4f}'.format(metrics['tce']))

    calibration_curve_path = args.input_path.replace('.json', '_calibration_curve.png')

    plot_calibration_emnlp23(
        all_scores, all_labels,
        output_path=calibration_curve_path,
        n_bins=10,
        prediction_mask=prediction_mask
    )
    print()
    print(f'Calibration curve saved to {calibration_curve_path}')


if __name__ == '__main__':
    main()
