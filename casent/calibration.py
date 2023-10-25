import accelerate
from typing import Tuple, Dict

import torch
from tqdm import tqdm
from casent.entity_typing_t5 import *
from casent.dataset import EntityTypingExample
from casent.utils.visualization_utils import *


def get_all_scores_labels(
        predictor: T5ForEntityTypingPredictor,
        dataloader,
        type_vocab: List[str],
        do_calibration: bool = True,
        do_thresholding: bool = True,
        do_constraint_beam_search: bool = True,
        num_beams: Optional[int] = None,
        num_decode: Optional[int] = None,
        progress_bar: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        all_scores: np.ndarray of shape (n_samples, n_types)
        all_labels: np.ndarray of shape (n_samples, n_types)
    """
    n_samples = len(dataloader.dataset)
    n_types = len(type_vocab)
    all_scores = np.full((n_samples, n_types), -np.inf, dtype=float)
    all_labels = np.zeros((n_samples, n_types), dtype=float)
    type2idx = {t: i for i, t in enumerate(type_vocab)}
    i = 0
    for batch in tqdm(dataloader, disable=not progress_bar):
        predictions = predictor.predict_batch(
            **batch,
            do_calibration=do_calibration,
            do_thresholding=do_thresholding,
            do_constraint_beam_search=do_constraint_beam_search,
            num_beams=num_beams,
            num_decode=num_decode
        )
        j = i + batch['input_ids'].size(0)
        for k, pred in enumerate(predictions):
            if len(pred.types) > 0:
                type_ids = np.array([type2idx[t] for t in pred.types], dtype=int)
                all_scores[i + k, type_ids] = pred.scores
        for k, types in enumerate(batch['raw_labels']):
            if len(types) > 0:
                type_ids = np.array([type2idx[t] for t in types], dtype=int)
                all_labels[i + k, type_ids] = 1.
        i = j
    assert j == n_samples and (all_labels.sum(1) > 0).all()
    return all_scores, all_labels


def get_all_types_prior(
        predictor: T5ForEntityTypingPredictor,
        tokenizer,
        type_vocab: List[str],
        eval_batch_size: int = 32
) -> np.ndarray:
    """
    Returns:
        scores: np.ndarray of shape (n_types,)
    """
    n_types = len(type_vocab)
    scores = np.zeros((n_types,), dtype=float)
    for i in range(0, len(type_vocab), eval_batch_size):
        j = min(i + eval_batch_size, len(type_vocab))
        pred, = predictor.score_raw(['<M></M>'], [type_vocab[i:j]], do_calibration=False)
        scores[i:j] = pred.scores
    return scores


def get_group_idx_by_freq_cutoff(
        all_labels: np.ndarray,
        cutoffs: List[int]
) -> List[np.ndarray]:
    """
    Returns:
        group_idx: list of [len(cutoffs) + 1] integer arrays
    """
    _, n_types = all_labels.shape
    group_idx = []
    low = 0
    for high in cutoffs + [np.inf]:
        group_mask = (all_labels.sum(0) >= low) & (all_labels.sum(0) < high)
        group_idx.append(np.nonzero(group_mask)[0])
        low = high
    assert np.concatenate(group_idx).shape[0] == n_types
    return group_idx


def get_group_idx(
        type_vocab: List[str],
        type_freq: Dict[str, int],
        cutoffs: List[int]
) -> List[np.ndarray]:
    """
    Returns:
        group_idx: list of [len(cutoffs) + 1] integer arrays
    """
    n_types = len(type_vocab)
    group_idx = []
    low = 0
    for high in cutoffs + [np.inf]:
        group_idx.append(np.array([i for i, t in enumerate(type_vocab)
                                   if low <= type_freq[t] < high]))
        low = high
    assert np.concatenate(group_idx).shape[0] == n_types
    return group_idx


def ufet_f1_array(all_preds: np.ndarray,
                  all_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    all_preds: array of shape (*, n_samples, n_labels)
    all_labels: array of shape (*, n_samples, n_labels)

    Returns:
        p:  array of shape (*)
        r:  array of shape (*)
        f1:  array of shape (*)
    """
    assert all_preds.shape[-2:] == all_labels.shape[-2:]
    common = (all_preds * all_labels).sum(-1)  # (*, n_samples)
    pred_nums = all_preds.sum(-1)  # (*, n_samples)
    p = common / np.maximum(pred_nums, 1e-20)  # (*, n_samples)
    p = p.sum(-1) / np.maximum((pred_nums > 0).sum(-1), 1e-20)  # (*)
    assert (all_labels.sum(-1) > 0).all()
    r = (common / all_labels.sum(-1)).mean(-1)
    f1 = 2 * p * r / np.maximum(p + r, 1e-20)
    return p, r, f1


def grid_search_group_threshold(
        all_scores: np.ndarray,
        all_labels: np.ndarray,
        group_idx: List[np.ndarray],
        n_grids=10
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Returns:
        threshold: np.ndarray of shape (n_types,)
        best_result: dict
    """
    n_groups = len(group_idx)
    n_samples, n_types = all_labels.shape

    bs = 16
    all_trange = []
    for idx in group_idx:
        mask = all_scores[:, idx] > -np.inf
        if mask.any():
            m = all_scores[:, idx][mask].min()
            M = all_scores[:, idx][mask].max()
            r = np.linspace(m, M, n_grids)
            all_trange.append(r)
        else:
            all_trange.append(np.array([-np.inf]))
    candidates = np.stack([x.flatten() for x in np.meshgrid(*all_trange)], 1)
    assert candidates.shape == (len(all_trange[0]) ** n_groups, n_groups), candidates.shape
    group_threshold = None
    best_result = {}
    threshold_buf = np.zeros((bs, n_types), dtype=float)
    pred_buf = np.zeros((bs, n_samples, n_types), dtype=bool)
    for a in range(0, candidates.shape[0], bs):
        b = min(a + bs, candidates.shape[0])
        batch_ts = candidates[a:b]  # (bs, n_groups)
        for gi, idx in enumerate(group_idx):
            threshold_buf[:b - a, idx] = batch_ts[:, gi, None]
        np.greater(all_scores, threshold_buf[:b - a, None, :], out=pred_buf[:b - a])
        p, r, f1 = ufet_f1_array(pred_buf[:b - a], all_labels)
        if group_threshold is None or f1.max() > best_result['f']:
            best_idx = f1.argmax()
            best_result = {
                'p': float(p[best_idx]),
                'r': float(r[best_idx]),
                'f': float(f1[best_idx])
            }
            group_threshold = batch_ts[best_idx]
    threshold = np.zeros((n_types,), dtype=float)
    for gi, idx in enumerate(group_idx):
        threshold[idx] = group_threshold[gi]
    return threshold, best_result


def distributed_calibrate_single_threshold(
        accelerator: accelerate.Accelerator,
        predictor: T5ForEntityTypingPredictor,
        dataloader,
        type_vocab: List[str],
        n_grids: int = 200
) -> Dict[str, float]:
    """
    Returns:
        metrics: dict
    """
    assert isinstance(predictor.model.calibration, ThresholdCalibration)

    if accelerator.is_main_process:
        all_scores, all_labels = get_all_scores_labels(predictor, dataloader,
                                                       type_vocab, do_calibration=False)
        threshold, metrics = grid_search_group_threshold(
            all_scores, all_labels,
            group_idx=[np.arange(len(type_vocab))],
            n_grids=n_grids)
        threshold = torch.tensor(threshold)
    else:
        threshold = None
        metrics = None
    accelerator.wait_for_everyone()
    threshold, metrics = accelerate.utils.broadcast_object_list([threshold, metrics])
    predictor.model.calibration.update_params(threshold=threshold)
    return metrics


def distributed_calibrate_platt_prior(
        accelerator: accelerate.Accelerator,
        predictor: T5ForEntityTypingPredictor,
        dataloader,
        type_vocab: List[str],
        type_freq: Dict[str, int],
        tokenizer,
        n_grids: int = 200,
        with_constrained_beam_search: bool = False
) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    assert isinstance(predictor.model.calibration, PriorPlattCalibration)

    if accelerator.is_main_process:
        cutoffs = [2, 4, 8, 16, 32, 64, 128, 256]
        all_scores, all_labels = get_all_scores_labels(
            predictor, dataloader, type_vocab,
            do_calibration=False,

            # important: do not run constraint beam search when estimating calibration parameters
            do_constraint_beam_search=with_constrained_beam_search
        )
        n_samples, n_types = all_labels.shape

        prior = get_all_types_prior(predictor, tokenizer, type_vocab)
        # group_idx = get_group_idx(type_vocab, type_freq, cutoffs)
        group_idx = get_group_idx_by_freq_cutoff(all_labels, cutoffs)
        weights = np.zeros((n_types, 2), dtype=float)
        bias = np.zeros((n_types,), dtype=float)
        weights[:, 0] = 1.0
        prior_expanded = np.broadcast_to(prior, (n_samples, n_types))

        # note: we only sample from labels with at least one positive example
        #       and with both positive and negative prediction
        global_label_mask = (((all_labels > 0) & (all_scores > -np.inf)).sum(0) > 0) \
                            & (((all_labels == 0) & (all_scores > -np.inf)).sum(0) > 0)
        global_pos_mask = (all_scores > -np.inf) & (all_labels > 0) & global_label_mask
        global_neg_mask = (all_scores > -np.inf) & (all_labels == 0) & global_label_mask
        # print((all_scores > -np.inf).sum())
        # print(global_pos_mask.sum(), global_neg_mask.sum())

        # some groups might not have any positive/negative example
        # in this case, we use merge them into the next group
        acc_idx = None
        acc_X_train, acc_y_train = None, None

        for gi, idx in enumerate(group_idx):
            label_mask = np.zeros((n_types,), dtype=bool)
            label_mask[idx] = True
            pos_mask = global_pos_mask & label_mask
            neg_mask = global_neg_mask & label_mask
            X_scores = np.concatenate((all_scores[pos_mask], all_scores[neg_mask]), axis=0)
            X_prior = np.concatenate((prior_expanded[pos_mask], prior_expanded[neg_mask]), axis=0)
            X_train = np.stack((X_scores, X_prior), axis=1)
            y_train = np.array([1] * pos_mask.sum() + [0] * neg_mask.sum())
            assert X_train.shape[0] == y_train.shape[0]

            if acc_idx is not None:
                idx = np.concatenate((idx, acc_idx), axis=0)
                X_train = np.concatenate((X_train, acc_X_train), axis=0)
                y_train = np.concatenate((y_train, acc_y_train), axis=0)

            if (y_train == 0).sum() == 0 or (y_train == 1).sum() == 0:
                acc_idx = idx
                acc_X_train = X_train
                acc_y_train = y_train
                continue

            # print(gi, idx.shape, pos_mask.sum(), neg_mask.sum())

            clf = LogisticRegression(random_state=0, fit_intercept=True,
                                     class_weight=None, penalty=None)
            clf.fit(X_train, y_train)
            weights[idx] = clf.coef_
            bias[idx] = clf.intercept_

            acc_idx = None
            acc_X_train, acc_y_train = None, None

        if acc_idx is not None:
            print(f'Warning: no calibration performed for {len(acc_idx)} types')

        with np.errstate(over='ignore'):
            all_scores_calibrated = 1 / (1 + np.exp(-(weights[:, 0] * np.maximum(all_scores, -1e20)
                                                      + weights[:, 1] * prior + bias)))

        threshold, metrics = grid_search_group_threshold(all_scores_calibrated, all_labels,
                                                         [np.arange(n_types)], n_grids=n_grids)
        print(metrics)
    else:
        prior, weights, bias, threshold = None, None, None, None
        metrics = None
    accelerator.wait_for_everyone()
    prior, weights, bias, threshold, metrics = accelerate.utils.broadcast_object_list(
        [prior, weights, bias, threshold, metrics])
    predictor.model.calibration.update_params(
        prior=torch.from_numpy(prior),
        weights=torch.from_numpy(weights),
        bias=torch.from_numpy(bias),
        threshold=torch.from_numpy(threshold)
    )
    return metrics


def calibrate_platt_prior_single_label(
        predictor: T5ForEntityTypingPredictor,
        dev_entity_mentions: List[str],
        dev_labels: List[str],
        type_vocab: List[str],
        eval_batch_size: int = 8,
) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    assert isinstance(predictor.model.calibration, PriorPlattCalibration)

    n_samples = len(dev_entity_mentions)
    n_types = len(type_vocab)
    all_scores = np.full((n_samples, n_types), -np.inf, dtype=float)
    all_labels = np.zeros((n_samples, n_types), dtype=int)
    type2idx = {t: i for i, t in enumerate(type_vocab)}

    for i, pred in enumerate(predictor.score_raw(
            dev_entity_mentions,
            [type_vocab] * n_samples,
            eval_batch_size=eval_batch_size,
            do_calibration=False,
            progress_bar=True
    )):
        all_scores[i] = np.array(pred.scores)
        all_labels[i, type2idx[dev_labels[i]]] = 1

    prior = get_all_types_prior(predictor, predictor.tokenizer, type_vocab)
    weights = np.zeros((n_types, 2), dtype=float)
    bias = np.zeros((n_types,), dtype=float)

    for k in range(n_types):
        X_scores = all_scores[:, k]
        X_prior = np.array([prior[k]] * n_samples)  # reduces to platt scaling
        X_train = np.stack((X_scores, X_prior), axis=1)
        y_train = all_labels[:, k]
        assert X_train.shape[0] == y_train.shape[0]

        clf = LogisticRegression(random_state=0, fit_intercept=True,
                                 class_weight=None, penalty=None)
        clf.fit(X_train, y_train)
        weights[k] = clf.coef_
        bias[k] = clf.intercept_

    with np.errstate(over='ignore'):
        all_scores_calibrated = 1 / (1 + np.exp(-(weights[:, 0] * np.maximum(all_scores, -1e20)
                                                  + weights[:, 1] * prior + bias)))

    accuracy = (all_scores_calibrated.argmax(1) == all_labels.argmax(1)).mean()

    predictor.model.calibration.update_params(
        prior=torch.from_numpy(prior),
        weights=torch.from_numpy(weights),
        bias=torch.from_numpy(bias),
        threshold=torch.full((n_types,), -np.inf)
    )
    return {'accuracy': accuracy}


def calibrate(
        accelerator: accelerate.Accelerator,
        predictor: T5ForEntityTypingPredictor,
        dataloader,
        type_vocab: List[str],
        type_freq: Dict[str, int],
        strategy='disabled',
        tokenizer=None,
        with_constrained_beam_search: bool = False
) -> Optional[Dict[str, float]]:
    """
    Calibrate per-label prediction threshold by maximizing per-label f1 score
    """
    if strategy == 'disabled':
        metrics = None
    elif strategy == 'single':
        metrics = distributed_calibrate_single_threshold(
            accelerator, predictor, dataloader, type_vocab)
    elif strategy == 'prior_platt':
        metrics = distributed_calibrate_platt_prior(
            accelerator, predictor, dataloader, type_vocab, type_freq, tokenizer,
            with_constrained_beam_search=with_constrained_beam_search
        )
    else:
        raise ValueError(f'Unknown calibration strategy {strategy}')
    return metrics


def plot_ufet_group_calibration_curves(
        all_scores: np.ndarray, all_labels: np.ndarray,
        output_path: str = 'tmp/cali_curve_dev.png',
        n_bins: int = 5,
        prediction_mask: Optional[np.ndarray] = None,
):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    mask = all_scores > -np.inf

    if prediction_mask is not None:
        mask = mask & prediction_mask

    assert ((all_scores[mask] >= 0) & (all_scores[mask] <= 1)).all()

    n_types = all_labels.shape[1]

    cutoffs = [5, 10, 20, 50, 100]
    # cutoffs = [10, 50]
    group_idx = get_group_idx_by_freq_cutoff(all_labels, cutoffs)

    n_groups = len(group_idx)
    n_col = 3
    n_row = (n_groups - 1) // 3 + 1
    gs = GridSpec(n_row * 3, n_col)
    fig = plt.Figure(figsize=(n_col * 3, n_row * 4))

    for gi, idx in enumerate(group_idx):
        pos_x = gi // 3
        pos_y = gi % 3
        ax = fig.add_subplot(gs[pos_x * 3: pos_x * 3 + 2, pos_y])

        label_mask = np.zeros((n_types,), dtype=bool)
        label_mask[idx] = True

        prob_true, prob_pred = calibration_curve(all_labels[mask & label_mask],
                                                 all_scores[mask & label_mask],
                                                 n_bins=n_bins)
        low = 0 if gi == 0 else cutoffs[gi - 1]
        high = np.inf if gi == len(group_idx) - 1 else cutoffs[gi]

        ax.plot(prob_pred, prob_true, color=COLORS['mint'], label=f'[{low}, {high})')
        ax.scatter(prob_pred, prob_true, color=COLORS['mint'], zorder=3)

        ax.legend()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        # ax.set_title(f'[{low}, {high})')
        ax.grid()
        set_axis_style(ax)

        hist, edges = np.histogram(all_scores[:, idx].reshape(-1), bins=10, range=(0., 1.))
        # print(hist)

        hist = hist / hist.sum()
        y_max = hist[1:].max() * 1.2

        ax1 = fig.add_subplot(gs[pos_x * 3 + 2, pos_y])
        ax1.stairs(hist, edges, fill=True, color=COLORS['mint'], zorder=3)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, y_max)
        ax1.set_xticks(np.linspace(0, 1, 6))
        ax1.set_yticks([y_max])
        # ax1.ticklabel_format(axis='y', style='sci')
        set_axis_style(ax1)

    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    # fig.tight_layout()
    fig.savefig(output_path, format='png', bbox_inches='tight')


def plot_calibration_curves(
        all_scores: np.ndarray, all_labels: np.ndarray,
        output_path: str = 'tmp/cali_curve_dev.png',
        n_bins: int = 5,
):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    mask = all_scores > -np.inf
    assert ((all_scores[mask] >= 0) & (all_scores[mask] <= 1)).all()

    n_types = all_labels.shape[1]

    n_col = 1
    n_row = 1
    gs = GridSpec(n_row * 3, n_col)
    fig = plt.Figure(figsize=(3, 4))

    ax = fig.add_subplot(gs[0: 2, 0])

    prob_true, prob_pred = calibration_curve(all_labels[mask],
                                             all_scores[mask],
                                             n_bins=n_bins)

    ax.plot(prob_pred, prob_true, color=COLORS['mint'])
    ax.scatter(prob_pred, prob_true, color=COLORS['mint'], zorder=3)

    # ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    # ax.set_title(f'[{low}, {high})')
    ax.grid()
    set_axis_style(ax)

    hist, edges = np.histogram(all_scores[mask].reshape(-1), bins=10, range=(0., 1.))
    # print(hist)

    hist = hist / hist.sum()
    y_max = hist[1:].max() * 1.2

    ax1 = fig.add_subplot(gs[2, 0])
    ax1.stairs(hist, edges, fill=True, color=COLORS['mint'], zorder=3)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, y_max)
    ax1.set_xticks(np.linspace(0, 1, 6))
    ax1.set_yticks([y_max])
    # ax1.ticklabel_format(axis='y', style='sci')
    set_axis_style(ax1)

    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    # fig.tight_layout()
    fig.savefig(output_path, format='png', bbox_inches='tight')


def plot_calibration_emnlp23(
        all_scores: np.ndarray, all_labels: np.ndarray,
        output_path: str,
        n_bins: int = 5,
        prediction_mask: Optional[np.ndarray] = None,
):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    mask = all_scores > -np.inf

    if prediction_mask is not None:
        mask = mask & prediction_mask

    assert ((all_scores[mask] >= 0) & (all_scores[mask] <= 1)).all()

    n_types = all_labels.shape[1]

    cutoffs = [10]
    group_idx = get_group_idx_by_freq_cutoff(all_labels, cutoffs)

    n_groups = len(group_idx)
    fig = plt.Figure(figsize=(6, 3))

    for gi, idx in enumerate(group_idx):
        ax = fig.add_subplot(1, n_groups, gi + 1)

        label_mask = np.zeros((n_types,), dtype=bool)
        label_mask[idx] = True

        prob_true, prob_pred = calibration_curve(all_labels[mask & label_mask],
                                                 all_scores[mask & label_mask],
                                                 n_bins=n_bins)
        low = 0 if gi == 0 else cutoffs[gi - 1]
        high = np.inf if gi == len(group_idx) - 1 else cutoffs[gi]

        ax.plot([0., 1.], [0., 1.], '--', color='#b8b8b8', label=f'Perfect calibration')

        # color = '#c5e0b4'
        color = '#a4c98d'
        ax.plot(prob_pred, prob_true, color=color, label=f'CASENT')
        ax.scatter(prob_pred, prob_true, color=color, zorder=3)

        ax.legend()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="y", direction="in", pad=6)
        ax.tick_params(axis="x", direction="in", pad=6)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        if gi > 0:
            ax.set_yticklabels([])
        ax.set_xlabel('Confidence')
        if gi == 0:
            ax.set_ylabel('Accuracy')
        if gi == 0:
            ax.set_title(f'Calibration on rare types')
        else:
            ax.set_title(f'Calibration on frequent types')
        ax.grid(color='#d6d6d6')

        ax.spines['bottom'].set_color('#6e6e6e')
        ax.spines['top'].set_color('#6e6e6e')
        ax.spines['right'].set_color('#6e6e6e')
        ax.spines['left'].set_color('#6e6e6e')
        ax.tick_params(length=0)

    fig.subplots_adjust(wspace=0.12)
    # fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')


def adapt_platt_prior(
        predictor: T5ForEntityTypingPredictor,
        new_type_vocab: List[str],
) -> None:
    assert isinstance(predictor.model.calibration, PriorPlattCalibration)

    old_type_vocab = predictor.config.type_vocab

    if new_type_vocab == predictor.config.type_vocab:
        return

    new_prior = get_all_types_prior(predictor, predictor.tokenizer, new_type_vocab)
    new_prior = torch.from_numpy(new_prior)

    old_t2i = {t: i for i, t in enumerate(old_type_vocab)}
    oov_type_id = old_t2i[predictor.config.oov_type]
    new2old = torch.tensor([old_t2i.get(t, oov_type_id) for t in new_type_vocab])

    new_weights = predictor.model.calibration.weights[new2old]
    new_bias = predictor.model.calibration.bias[new2old]
    new_threshold = predictor.model.calibration.threshold[new2old]

    new_calibration = PriorPlattCalibration(
        type_vocab=new_type_vocab,
        oov_type=None,
        prior=new_prior,
        weights=new_weights,
        bias=new_bias,
        threshold=new_threshold,
    )

    predictor.update_calibration_module(
        new_calibration=new_calibration,
        new_type_vocab=new_type_vocab,
        new_oov_type=None
    )


def evaluate_calibration_error(all_scores: np.ndarray, all_labels: np.ndarray,
                               n_bins: int = 5, prediction_mask=None) -> Dict[str, float]:
    """
    Computes several metrics used to measure calibration error:
        - Expected Calibration Error (ECE): \sum_k (b_k / n) |acc(k) - conf(k)|
        - Maximum Calibration Error (MCE): max_k |acc(k) - conf(k)|
        - Total Calibration Error (TCE): \sum_k |acc(k) - conf(k)|
    """
    from sklearn.calibration import calibration_curve

    mask = all_scores > -np.inf

    if prediction_mask is not None:
        mask = mask & prediction_mask

    # print('[Calibration] Number of samples:', mask.sum())

    if not ((all_scores[mask] >= 0) & (all_scores[mask] <= 1)).all():
        return {
            'ece': np.nan,
            'mce': np.nan,
            'tce': np.nan,
        }

    true_label = all_labels[mask]
    confidence = all_scores[mask]

    prob_true, prob_pred = calibration_curve(true_label, confidence, n_bins=n_bins)
    bin_counts, _ = np.histogram(confidence, bins=np.linspace(0, 1, n_bins + 1))
    bin_counts = bin_counts[bin_counts > 0]
    ece = (np.abs(prob_true - prob_pred) * (bin_counts / len(true_label))).sum()
    mce = np.abs(prob_true - prob_pred).max()
    tce = np.abs(prob_true - prob_pred).sum()
    return {
        'ece': ece,
        'mce': mce,
        'tce': tce,
    }
