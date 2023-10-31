from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import PreTrainedTokenizer, BatchEncoding
from torch.utils.data import DataLoader
from datasets import Dataset
import torch
import torch.nn as nn
import random
from tqdm import trange, tqdm
from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Literal, Dict, Any
from casent.trie import Trie
from casent.dataset import UFETDataset, EntityTypingExample
from casent.entity_typing_common import EntityTypingOutput, AutoTokenizerForEntityTyping
from casent.utils.ner_utils import *
from casent.utils.wikidata_ontology import WikidataDAGOntology, WikidataConcept


class T5ForEntityTypingConfig(T5Config):
    """
    pred_num_decode: top N decoded token sequences are as predictions
    pred_num_beams: beam size for beam searching the top sequences
    """
    pred_num_decode: int = 12  # top N decoded token sequences are as predictions
    pred_num_beams: int = 24  # beam size for beam searching the top sequences
    entity_max_length: int = 128  # maximum token length for input entity mentions
    decode_prefix: str = 'In this sentence, {} is'

    # for calibration
    calibration: Literal['disabled', 'single', 'prior_platt'] = 'prior_platt'
    type_vocab: List[str] = []
    oov_type: Optional[str] = 'businesswoman'

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     if len(self.type_vocab) == 0:
    #         self.type_vocab = UFETDataset.get_type_vocab('data/ufet')


class IdentityCalibration(nn.Module):
    def forward(self, predictions: List[EntityTypingOutput]) -> List[EntityTypingOutput]:
        return predictions


class ThresholdCalibration(nn.Module):
    def __init__(self, type_vocab: List[str],
                 oov_type: Optional[str],
                 threshold: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer('threshold', torch.full((len(type_vocab),), -np.inf))
        self.type2idx = {t: i for i, t in enumerate(type_vocab)}
        self.oov_type_idx = self.type2idx[oov_type] if oov_type is not None else None
        if threshold is not None:
            self.update_params(threshold)

    def update_params(self, threshold: torch.Tensor):
        self.threshold[:] = threshold
        self._cache()

    def _cache(self):
        self._threshold = self.threshold.cpu().numpy()

    def forward(self, predictions: List[EntityTypingOutput]) -> List[EntityTypingOutput]:
        if not hasattr(self, '_threshold'):
            self._cache()

        res = []
        for pred in predictions:
            type_ids = np.array([self.type2idx.get(t, -1) for t in pred.types], dtype=int)
            oov_mask = type_ids < 0
            if oov_mask.any():
                if self.oov_type_idx is None:
                    raise ValueError('The model predicts an out-of-vocabulary type and'
                                     ' the calibration module does not have an `oov_type`')
                else:
                    mapped_type_ids = np.where(oov_mask, self.oov_type_idx, type_ids)
            else:
                mapped_type_ids = type_ids
            scores = np.array(pred.scores)
            mask = scores > self._threshold[mapped_type_ids]
            res.append(EntityTypingOutput(
                types=[t for i, t in enumerate(pred.types) if mask[i]],
                scores=scores[mask].tolist()
            ))
        return res


class PriorPlattCalibration(nn.Module):
    """
    p'(type|entity) = σ( w1 * log_p(type|entity) + w2 * log_p(type|Ø) + b )

    Args:
        type_vocab: list
        oov_type: str
        prior (optional): torch.Tensor of shape (num_entity_types,)
        weights (optional): torch.Tensor of shape (num_entity_types, 2)
        bias (optional): torch.Tensor of shape (num_entity_types,)
        threshold (optional): torch.Tensor of shape (num_entity_types,)
    """

    def __init__(self,
                 type_vocab: List[str],
                 oov_type: Optional[str],
                 prior: Optional[torch.Tensor] = None,
                 weights: Optional[torch.Tensor] = None,
                 bias: Optional[torch.Tensor] = None,
                 threshold: Optional[torch.Tensor] = None):
        super().__init__()

        self.type2idx = {t: i for i, t in enumerate(type_vocab)}
        self.oov_type_idx = self.type2idx[oov_type] if oov_type is not None else None

        n_types = len(type_vocab)
        self.register_buffer('prior', torch.full((n_types,), 0.))
        self.register_buffer('weights', torch.full((n_types, 2), 0.))
        self.register_buffer('bias', torch.full((n_types,), 0.))
        self.register_buffer('threshold', torch.full((n_types,), -np.inf))
        self.update_params(prior, weights, bias, threshold)

    def update_params(self, prior: Optional[torch.Tensor] = None,
                      weights: Optional[torch.Tensor] = None,
                      bias: Optional[torch.Tensor] = None,
                      threshold: Optional[torch.Tensor] = None):
        if prior is not None:
            self.prior[:] = prior
        if weights is not None:
            self.weights[:] = weights
        if bias is not None:
            self.bias[:] = bias
        if threshold is not None:
            self.threshold[:] = threshold
        self._cache()

    def _cache(self):
        self._prior = self.prior.cpu().numpy()
        self._weights = self.weights.cpu().numpy()
        self._bias = self.bias.cpu().numpy()
        self._threshold = self.threshold.cpu().numpy()

    @staticmethod
    def _sigmoid(x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def forward(self, predictions: List[EntityTypingOutput]) -> List[EntityTypingOutput]:
        if not hasattr(self, '_prior'):
            self._cache()

        res = []
        for pred in predictions:
            type_ids = np.array([self.type2idx.get(t, -1) for t in pred.types], dtype=int)
            oov_mask = type_ids < 0
            if oov_mask.any():
                if self.oov_type_idx is None:
                    raise ValueError('The model predicts an out-of-vocabulary type and'
                                     ' the calibration module does not have an `oov_type`')
                else:
                    mapped_type_ids = np.where(oov_mask, self.oov_type_idx, type_ids)
            else:
                mapped_type_ids = type_ids

            scores = self._sigmoid(self._weights[mapped_type_ids, 0] * np.array(pred.scores)
                                   + self._weights[mapped_type_ids, 1] * self._prior[mapped_type_ids]
                                   + self._bias[mapped_type_ids])

            # type2prior = {t: self._prior[i] for t, i in zip(pred.types, mapped_type_ids)}
            # print(type2prior)

            mask = scores > self._threshold[mapped_type_ids]
            res.append(EntityTypingOutput(
                types=[t for i, t in enumerate(pred.types) if mask[i]],
                scores=scores[mask].tolist()
            ))
        return res


class T5ForEntityTyping(T5ForConditionalGeneration):
    def __init__(self, config: T5ForEntityTypingConfig):
        super().__init__(config)

        assert all('_' not in t for t in config.type_vocab)
        # self.config.type_vocab = [t.replace('_', ' ') for t in self.config.type_vocab]

        if config.calibration == 'single':
            self.calibration = ThresholdCalibration(type_vocab=config.type_vocab, oov_type=config.oov_type)
        elif config.calibration == 'disabled':
            self.calibration = IdentityCalibration()
        elif config.calibration == 'prior_platt':
            self.calibration = PriorPlattCalibration(
                type_vocab=config.type_vocab, oov_type=config.oov_type
            )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            num_labels: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> Seq2SeqLMOutput:

        if labels is None:  # inference
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels, **kwargs
            )

        # training
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        for k, v in encoder_outputs.items():
            if v.size(0) == input_ids.size(0):
                encoder_outputs[k] = v.repeat_interleave(num_labels, dim=0)

        attention_mask = attention_mask.repeat_interleave(num_labels, dim=0)

        return super().forward(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels
        )


class T5ForEntityTypingPredictor:

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._update_trie()

        self.preprocess_fn = T5Preprocessor(tokenizer=self.tokenizer,
                                            max_length=self.config.entity_max_length,
                                            decode_prefix=self.config.decode_prefix)
        self.collate_fn = T5DataCollator(tokenizer=self.tokenizer)

    def update_calibration_module(
            self,
            new_calibration: nn.Module,
            new_type_vocab: List[str],
            new_oov_type: Optional[str]
    ):
        self.model.calibration = new_calibration
        for config in (self.config, self.model.config):
            config.type_vocab = new_type_vocab
            config.oov_type = new_oov_type
        self._update_trie()

    def _update_trie(self):
        self.max_type_length = max(self.tokenizer(self.config.type_vocab, return_length=True).length)
        self.trie = self._build_trie(self.tokenizer, self.config.type_vocab)

    @staticmethod
    def _build_trie(tokenizer, type_vocab: List[str]) -> Trie:
        token_sequences = tokenizer(
            [f'<pad>{t}' for t in type_vocab],
            padding='longest',
            return_tensors='pt'
        ).input_ids.tolist()
        return Trie(token_sequences)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        tokenizer = AutoTokenizerForEntityTyping.from_pretrained(pretrained_model_name_or_path)
        config = T5ForEntityTypingConfig.from_pretrained(pretrained_model_name_or_path)
        model = T5ForEntityTyping.from_pretrained(pretrained_model_name_or_path, config=config)
        return cls(model, tokenizer, config)

    def to(self, device):
        self.model.to(device)

    def _set_ninf_threshold(self) -> torch.Tensor:
        original_threshold = self.model.calibration.threshold.clone()
        threshold = original_threshold.clone()
        threshold[:] = -np.inf
        self.model.calibration.update_params(threshold=threshold)
        return original_threshold

    def _restore_threshold(self, original_threshold: torch.Tensor):
        self.model.calibration.update_params(threshold=original_threshold)

    def predict_batch(self, input_ids, attention_mask=None,
                      do_calibration: bool = True,
                      do_thresholding: bool = True,
                      do_constraint_beam_search: bool = True,
                      num_decode: Optional[int] = None,
                      num_beams: Optional[int] = None, **kwargs) -> List[EntityTypingOutput]:
        if num_decode is None:
            num_decode = self.config.pred_num_decode
        if num_beams is None:
            num_beams = self.config.pred_num_beams
        assert num_beams >= num_decode, f'num_beams({num_beams}) < num_decode({num_decode})'

        if do_constraint_beam_search:
            prefix_allowed_tokens_fn = lambda batch_id, sent: self.trie.get(sent.tolist())
        else:
            prefix_allowed_tokens_fn = None

        # print(self.tokenizer.decode(input_ids[0].tolist()))

        self.model.eval()
        with torch.no_grad():
            gen_outputs = self.model.generate(
                input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                num_return_sequences=num_decode,
                max_length=min(self.config.max_length, self.max_type_length),
                num_beams=num_beams,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )
        self.model.train()

        assert gen_outputs.sequences.size(0) == num_decode * input_ids.size(0)
        preds = self.tokenizer.batch_decode(gen_outputs.sequences, skip_special_tokens=True)
        preds = [preds[a:a + num_decode] for a in range(0, len(preds), num_decode)]
        scores = gen_outputs.sequences_scores.view(-1, num_decode)

        outputs = []
        for ts, ss in zip(preds, scores):
            type2score = {}
            for t, s in zip(ts, ss):
                if t in self.config.type_vocab:
                    type2score[t] = max(float(s), type2score.get(t, -float('inf')))
            outputs.append(EntityTypingOutput(
                types=list(type2score.keys()),
                scores=list(type2score.values())
            ))

        if do_calibration:
            if not do_thresholding and hasattr(self.model.calibration, 'threshold'):
                original_threshold = self._set_ninf_threshold()
                outputs = self.model.calibration(outputs)
                self._restore_threshold(original_threshold)
            else:
                outputs = self.model.calibration(outputs)

        for output in outputs:
            type2score = {t: s for t, s in zip(output.types, output.scores)}
            output.types = sorted(type2score.keys(), key=lambda t: -type2score[t])
            output.scores = [type2score[t] for t in output.types]

        assert len(outputs) == input_ids.size(0)
        return outputs

    def score_batch(self, input_ids, attention_mask, num_labels, labels,
                    raw_labels, do_calibration=True, **kwargs) -> List[EntityTypingOutput]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                num_labels=num_labels.to(self.model.device),
                labels=labels.to(self.model.device)
            )
        self.model.train()

        labels = labels.to(self.model.device)
        logits = outputs.logits
        assert logits.size(0) == labels.size(0)
        mask = labels != -100
        labels[labels == -100] = 0
        token_log_probs = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (bs, sl)
        token_log_probs -= logits.logsumexp(2)  # (bs, sl)
        label_log_probs = (token_log_probs * mask.float()).sum(1)  # (bs,)
        label_log_probs /= mask.float().sum(1)
        label_log_probs = label_log_probs.cpu().numpy().tolist()

        res = []
        offset = 0
        for types in raw_labels:
            res.append(EntityTypingOutput(
                types=types,
                scores=label_log_probs[offset: offset + len(types)]
            ))
            offset += len(types)

        if do_calibration:
            if hasattr(self.model.calibration, 'threshold'):
                original_threshold = self._set_ninf_threshold()
                res = self.model.calibration(res)
                self._restore_threshold(original_threshold)
            else:
                res = self.model.calibration(res)

        return res

    def predict_raw(self, entity_mentions: List[str],
                    do_calibration: bool = True,
                    num_decode: Optional[int] = None,
                    num_beams: Optional[int] = None,
                    eval_batch_size: int = 1,
                    progress_bar: bool = False) -> List[EntityTypingOutput]:
        res = []
        for i in trange(0, len(entity_mentions), eval_batch_size, disable=not progress_bar):
            j = min(i + eval_batch_size, len(entity_mentions))
            batch = tokenize_t5(self.tokenizer,
                                contexts=entity_mentions[i:j],
                                max_length=self.config.entity_max_length,
                                decode_prefix=self.config.decode_prefix)
            res += self.predict_batch(**batch, num_decode=num_decode, num_beams=num_beams,
                                      do_calibration=do_calibration, do_thresholding=True)
        return res

    def score_raw(self, entity_mentions: List[str],
                  labels: List[List[str]],
                  eval_batch_size: int = 1,
                  do_calibration: bool = True,
                  progress_bar: bool = False) -> List[EntityTypingOutput]:
        examples = [EntityTypingExample(id=None, context=s, labels=ts)
                    for s, ts in zip(entity_mentions, labels)]
        res = []
        for i in trange(0, len(entity_mentions), eval_batch_size, disable=not progress_bar):
            j = min(i + eval_batch_size, len(entity_mentions))
            tokenized = [self.preprocess_fn(ex) for ex in examples[i:j]]
            batch = self.collate_fn(tokenized)
            res += self.score_batch(**batch, do_calibration=do_calibration)
        return res


@dataclass
class WikidataEntityTypingOutput:
    wd_types: List[WikidataConcept]
    scores: List[float]


class T5ForWikidataEntityTypingPredictor:

    def __init__(self, model, tokenizer, config, ontology):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._update_trie()

        self.ontology = ontology
        self.ufet2wd = {t: c for c in ontology.concepts() for t in c.ufet_labels}

        self.preprocess_fn = T5Preprocessor(tokenizer=self.tokenizer,
                                            max_length=self.config.entity_max_length,
                                            decode_prefix=self.config.decode_prefix)
        self.collate_fn = T5DataCollator(tokenizer=self.tokenizer)

    def update_calibration_module(
            self,
            new_calibration: nn.Module,
            new_type_vocab: List[str],
            new_oov_type: Optional[str]
    ):
        self.model.calibration = new_calibration
        for config in (self.config, self.model.config):
            config.type_vocab = new_type_vocab
            config.oov_type = new_oov_type
        self._update_trie()

    def _update_trie(self):
        self.max_type_length = max(self.tokenizer(self.config.type_vocab, return_length=True).length)
        self.trie = self._build_trie(self.tokenizer, self.config.type_vocab)

    @staticmethod
    def _build_trie(tokenizer, type_vocab: List[str]) -> Trie:
        token_sequences = tokenizer(
            [f'<pad>{t}' for t in type_vocab],
            padding='longest',
            return_tensors='pt'
        ).input_ids.tolist()
        return Trie(token_sequences)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            ontology_path: str,
    ):
        tokenizer = AutoTokenizerForEntityTyping.from_pretrained(pretrained_model_name_or_path)
        config = T5ForEntityTypingConfig.from_pretrained(pretrained_model_name_or_path)
        model = T5ForEntityTyping.from_pretrained(pretrained_model_name_or_path, config=config)
        ontology = WikidataDAGOntology.from_file(ontology_path)
        return cls(model, tokenizer, config, ontology)

    def to(self, device):
        self.model.to(device)

    def _set_ninf_threshold(self) -> torch.Tensor:
        original_threshold = self.model.calibration.threshold.clone()
        threshold = original_threshold.clone()
        threshold[:] = -np.inf
        self.model.calibration.update_params(threshold=threshold)
        return original_threshold

    def _restore_threshold(self, original_threshold: torch.Tensor):
        self.model.calibration.update_params(threshold=original_threshold)

    def predict_batch(self, input_ids, attention_mask=None,
                      do_calibration: bool = True,
                      do_thresholding: bool = True,
                      do_constraint_beam_search: bool = True,
                      num_decode: Optional[int] = None,
                      num_beams: Optional[int] = None,
                      include_parents: bool = False,
                      **kwargs) -> List[EntityTypingOutput]:
        if num_decode is None:
            num_decode = self.config.pred_num_decode
        if num_beams is None:
            num_beams = self.config.pred_num_beams
        assert num_beams >= num_decode, f'num_beams({num_beams}) < num_decode({num_decode})'

        if do_constraint_beam_search:
            prefix_allowed_tokens_fn = lambda batch_id, sent: self.trie.get(sent.tolist())
        else:
            prefix_allowed_tokens_fn = None

        # print(self.tokenizer.decode(input_ids[0].tolist()))

        self.model.eval()
        with torch.no_grad():
            gen_outputs = self.model.generate(
                input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                num_return_sequences=num_decode,
                max_length=min(self.config.max_length, self.max_type_length),
                num_beams=num_beams,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )
        self.model.train()

        assert gen_outputs.sequences.size(0) == num_decode * input_ids.size(0)
        preds = self.tokenizer.batch_decode(gen_outputs.sequences, skip_special_tokens=True)
        preds = [preds[a:a + num_decode] for a in range(0, len(preds), num_decode)]
        scores = gen_outputs.sequences_scores.view(-1, num_decode)

        outputs = []
        for ts, ss in zip(preds, scores):
            type2score = {}
            for t, s in zip(ts, ss):
                if t in self.config.type_vocab:
                    type2score[t] = max(float(s), type2score.get(t, -float('inf')))
            outputs.append(EntityTypingOutput(
                types=list(type2score.keys()),
                scores=list(type2score.values())
            ))

        if do_calibration:
            if not do_thresholding and hasattr(self.model.calibration, 'threshold'):
                original_threshold = self._set_ninf_threshold()
                outputs = self.model.calibration(outputs)
                self._restore_threshold(original_threshold)
            else:
                outputs = self.model.calibration(outputs)

        res = []
        for pred in outputs:
            wd_types = []
            qid2score = {}
            for t, s in zip(pred.types, pred.scores):

                if t not in self.ufet2wd:
                    continue

                base_concept = self.ufet2wd[t]
                for c in [base_concept] + (self.ontology.all_parents(base_concept) if include_parents else []):
                    if c.wd_qid not in qid2score:
                        wd_types.append(c)
                        qid2score[c.wd_qid] = s
                    else:
                        qid2score[c.wd_qid] = max(qid2score[c.wd_qid], s)

            wd_types = sorted(wd_types, key=lambda c: -qid2score[c.wd_qid])

            res.append(WikidataEntityTypingOutput(
                wd_types=wd_types,
                scores=[qid2score[c.wd_qid] for c in wd_types]
            ))
        assert len(res) == input_ids.size(0)
        return res

    def predict_raw(self, entity_mentions: List[str],
                    do_calibration: bool = True,
                    num_decode: Optional[int] = None,
                    num_beams: Optional[int] = None,
                    eval_batch_size: int = 1,
                    include_parents: bool = False,
                    progress_bar: bool = False) -> List[EntityTypingOutput]:
        res = []
        for i in trange(0, len(entity_mentions), eval_batch_size, disable=not progress_bar):
            j = min(i + eval_batch_size, len(entity_mentions))
            batch = tokenize_t5(self.tokenizer,
                                contexts=entity_mentions[i:j],
                                max_length=self.config.entity_max_length,
                                decode_prefix=self.config.decode_prefix)
            res += self.predict_batch(
                **batch, num_decode=num_decode, num_beams=num_beams,
                do_calibration=do_calibration, do_thresholding=True,
                include_parents=include_parents
            )
        return res


def extract_noun_phrases(
        root: stanza.models.constituency.parse_tree.Tree
) -> List[Tuple[int, int, str]]:
    """
    Extract a list of (start_token_idx, end_token_idx) that represents
    noun phrases. We only extract maximal noun phrases that have no
    sentence constituents inside.
    """
    NP = []
    S = []

    def dfs(curr_node):
        nonlocal start_token
        if curr_node.is_leaf():
            start_token += 1
        else:
            if curr_node.label == 'NP':
                NP.append((start_token, start_token + len(curr_node.leaf_labels())))
            elif curr_node.label == 'S':
                S.append((start_token, start_token + len(curr_node.leaf_labels())))
            for child in curr_node.children:
                dfs(child)

    start_token = 0
    dfs(root)

    NP = [(l, r) for l, r in NP if not any(l <= ll and r >= rr for ll, rr in S)]
    NP = [(l, r) for l, r in NP if not any(l >= ll and r <= rr for ll, rr in NP if (l, r) != (ll, rr))]
    return [(l, r, 'NP') for l, r in NP]


@dataclass
class EntityMentionSpan:
    start_char: int
    end_char: int  # mention = text[start_char:end_char]
    mention_span: str
    score: float
    all_types: List[str]
    all_scores: List[float]

    def __str__(self):
        return f'Mention("{self.mention_span}", {self.score:.2f})'

    def __repr__(self):
        return str(self)


def extract_entities_by_type(
        predictor: T5ForEntityTypingPredictor,
        stanza_pipeline: stanza.Pipeline,
        text: str,
        target_ufet_type: str,
        threshold: Optional[float] = None,
        eval_batch_size: int = 1,
        use_gpu: bool = False
) -> List[EntityMentionSpan]:
    if threshold is not None:
        raise NotImplementedError('threshold is not supported yet')

    if use_gpu:
        predictor.to(torch.device('cuda:0'))

    doc = stanza_pipeline(text)

    entity_typing_inputs = []
    spans = []
    for i, sentence in enumerate(doc.sentences):
        for start_token, end_token, label in extract_noun_phrases(sentence.constituency):
            start_char = sentence.tokens[start_token].start_char
            end_char = sentence.tokens[end_token - 1].end_char
            span = text[start_char:end_char].strip()
            # print(span, start_char, end_char)
            entity_typing_inputs.append(text[:start_char] + ' <M> ' + span + ' </M> ' + text[end_char:])
            spans.append(EntityMentionSpan(
                start_char=start_char,
                end_char=end_char,
                mention_span=span,
                score=-1.0,
                all_types=[],
                all_scores=[]
            ))

    preds = predictor.predict_raw(entity_typing_inputs, eval_batch_size=eval_batch_size)
    res = []
    for span, pred in zip(spans, preds):
        type2score = {t: s for t, s in zip(pred.types, pred.scores)}
        if target_ufet_type in type2score and (threshold is None or type2score[target_ufet_type] >= threshold):
            span.score = type2score[target_ufet_type]
            span.all_types = pred.types
            span.all_scores = pred.scores
            res.append(span)
    return res


def tokenize_t5(
        tokenizer: PreTrainedTokenizer,
        contexts: List[str],
        decode_prefix: str,
        max_length: Optional[int] = 128,
) -> BatchEncoding:
    assert '<M>' in tokenizer.vocab and '</M>' in tokenizer.vocab

    mentions = [s[s.index('<M>') + 3: s.index('</M>')].strip() for s in contexts]

    if decode_prefix != '':
        prefixes = [decode_prefix.format(m) if '{}' in decode_prefix else decode_prefix
                    for m in mentions]
    else:
        prefixes = None

    return tokenizer(
        contexts,
        prefixes,
        padding='longest',
        truncation='longest_first',
        add_special_tokens=True,
        return_tensors='pt',
        max_length=max_length,
        pad_to_multiple_of=8
    )


@dataclass
class T5Preprocessor:
    tokenizer: PreTrainedTokenizer
    decode_prefix: str
    max_length: int = 128

    def __call__(self, example) -> Dict:
        return {
            'id': example.id,
            'context': example.context,
            'input_ids': tokenize_t5(self.tokenizer, contexts=[example.context],
                                     max_length=self.max_length,
                                     decode_prefix=self.decode_prefix).input_ids[0],
            'raw_labels': example.labels,
            'labels': [self.tokenizer(t, return_tensors='pt').input_ids[0] for t in example.labels]
        }


@dataclass
class T5DataCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, samples: List[dict]) -> Dict:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [d['input_ids'] for d in samples], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [ids for d in samples for ids in d['labels']], batch_first=True, padding_value=-100
        )
        return {
            'id': [d['id'] for d in samples],
            'context': [d['context'] for d in samples],
            'input_ids': input_ids,
            'attention_mask': input_ids != self.tokenizer.pad_token_id,
            'labels': labels,
            'num_labels': torch.tensor([len(d['labels']) for d in samples]),
            'raw_labels': [d['raw_labels'] for d in samples],
        }


def subsample(samples: List[Any], size: Optional[int], split):
    if split != 'train' or size is None:
        return samples
    else:
        return random.sample(samples, size)


def get_dataloaders_t5(
        dataset_name, tokenizer, decode_prefix: str,
        ufet_dir='data/ufet/', max_length=128, batch_size=16, eval_batch_size=32,
        splits=('train', 'dev', 'test'),
        train_subsample_ratio: Optional[float] = None,
        train_subsample_size: Optional[int] = None,
        seed: int = 0,
        train_with_dev: bool = False,
        subsample_dev: float = -1.0,
) -> List[DataLoader]:
    assert train_subsample_ratio is None or train_subsample_size is None

    if dataset_name == 'ufet':
        if train_subsample_ratio is not None or train_subsample_size is not None:
            raise NotImplementedError()
        raw_examples = {
            split: list(UFETDataset(split, ufet_dir)) for split in splits
        }
        if train_with_dev:
            raw_examples['train'] += raw_examples['dev']
        if subsample_dev > 0.0:
            raw_examples['dev'] = random.sample(raw_examples['dev'], int(len(raw_examples['dev']) * subsample_dev))
    elif dataset_name in ('jnlpba', 'bc5cdr', 'mit_restaurant', 'mit_movie', 'wnut17'):
        label_mapping = TYPE_MAPPING[dataset_name]
        raw_examples = {split: load_ner_examples(dataset_name, split=split) for split in splits}
        for split in splits:
            ner_examples = raw_examples[split]
            entity_mentions, all_labels = prepare_typing_examples(ner_examples)

            if split == 'train':
                if train_subsample_ratio is not None:
                    entity_mentions, all_labels = subsample_by_ratio(
                        entity_mentions, all_labels,
                        ratio=train_subsample_ratio, seed=seed)
                elif train_subsample_size is not None:
                    entity_mentions, all_labels = subsample_by_size(
                        entity_mentions, all_labels,
                        size=train_subsample_size, seed=seed)

            raw_examples[split] = [
                EntityTypingExample(
                    id=None,
                    context=ent,
                    labels=[label_mapping[label]]
                )
                for ent, label in zip(entity_mentions, all_labels)
            ]
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    preprocess_fn = T5Preprocessor(tokenizer=tokenizer, max_length=max_length,
                                   decode_prefix=decode_prefix)
    collate_fn = T5DataCollator(tokenizer=tokenizer)

    dataloaders = []
    for split in splits:
        tokenized = [preprocess_fn(ex) for ex in tqdm(raw_examples[split], desc='tokenizing')]
        dataloaders.append(DataLoader(
            Dataset.from_list(tokenized).with_format("torch"),
            shuffle=(split == 'train'),
            batch_size=batch_size if split == 'train' else eval_batch_size,
            collate_fn=collate_fn
        ))
    return dataloaders
