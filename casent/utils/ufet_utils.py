import collections
from tqdm import tqdm
from typing import List, Any


def ufet_f1(all_preds: List[List[Any]], all_labels: List[List[Any]]):
    """Compute macro F1 score for multi-label classification tasks"""
    assert len(all_preds) == len(all_labels)
    p = 0.
    r = 0.
    n_pred = 0  # number of examples that has at least one prediction
    n_label = 0  # number of examples that has at least one label
    for preds, labels in zip(all_preds, all_labels):
        if len(preds) > 0:
            p += len(set(preds) & set(labels)) / len(set(preds))
            n_pred += 1
        if len(labels):
            r += len(set(preds) & set(labels)) / len(set(labels))
            n_label += 1
    p = p / n_pred if n_pred > 0 else 0.
    r = r / n_label if n_label > 0 else 0.
    f = 2 * p * r / (p + r) if r > 0 else 0.
    return p, r, f


def evaluate_metrics(predictor, dataloader, predict_kwargs={}) -> dict:
    all_preds = []
    all_labels = []
    for batch in dataloader:
        all_preds += [pred.types for pred in predictor.predict_batch(**batch, **predict_kwargs)]
        all_labels += batch['raw_labels']
    p, r, f = ufet_f1(all_preds, all_labels)
    result = {'p': p, 'r': r, 'f': f}
    return result


def evaluate_group_metrics(predictor, dataloader, predict_kwargs={}, cutoffs=[10, 50],
                           progress_bar: bool = False) -> List[dict]:
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, disable=not progress_bar):
        all_preds += [pred.types for pred in predictor.predict_batch(**batch, **predict_kwargs)]
        all_labels += batch['raw_labels']
    label_freq = collections.Counter(sum(all_labels, []))
    low = 0
    res = []
    for high in cutoffs + [float('inf')]:
        all_preds_group = [[t for t in ts if low <= label_freq[t] < high] for ts in all_preds]
        all_labels_group = [[t for t in ts if low <= label_freq[t] < high] for ts in all_labels]
        p, r, f = ufet_f1(all_preds_group, all_labels_group)
        res.append({'p': p, 'r': r, 'f': f})
        low = high
    p, r, f = ufet_f1(all_preds, all_labels)
    res.append({'p': p, 'r': r, 'f': f})
    return res
