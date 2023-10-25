import random
import argparse
import json
import time
import requests
import os
from transformers import RobertaForSequenceClassification, RobertaConfig, T5Tokenizer
from transformers import AutoTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple, Dict
from datasets import load_dataset
from casent.entity_typing_t5 import *
from casent.calibration import adapt_platt_prior, calibrate_platt_prior_single_label
from casent.utils.llm_backend import few_shot_query
from casent.utils.ner_utils import *


class roberta_mnli_typing(nn.Module):
    def __init__(self):
        super(roberta_mnli_typing, self).__init__()
        self.roberta_module = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
        self.config = RobertaConfig.from_pretrained("roberta-large-mnli")

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta_module(input_ids, attention_mask)
        res = nn.functional.softmax(roberta_output.logits, dim=-1)
        return res


def benchmark_lite(
        entity_mentions: List[str], type_vocab: List[str],
):
    model = roberta_mnli_typing()

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

    chkpt_path = 'checkpoints/lite/epochs1650_batch16_margin0.1_lr5e-06_Sep_09_2021.1_lr5e-06_15_27_30_Sep_09_2021'
    chkpt = torch.load(chkpt_path, map_location='cpu')
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    inference_time = []
    memory_usage = []
    for i, mention in enumerate(tqdm(entity_mentions)):
        to = time.time()
        premise = mention.replace('<M>', '').replace('</M>', '')
        entity = mention[mention.index('<M>') + 3:mention.index('</M>')]
        all_confidence = []
        for a in range(0, len(type_vocab), 32):
            b = min(a + 32, len(type_vocab))
            batch_types = type_vocab[a:b]
            sequence = [f'{premise}{2 * tokenizer.sep_token}{entity} is a {label}.' for label in batch_types]
            inputs = tokenizer(
                sequence, padding=True, return_tensors='pt', max_length=128
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)[:, -1]
            confidence = outputs.detach().cpu().numpy().tolist()
            all_confidence.extend(confidence)
        assert len(all_confidence) == len(type_vocab)
        type2score = {label: score for label, score in zip(type_vocab, all_confidence)}
        best_predicted = max(type2score, key=type2score.get)
        inference_time.append(time.time() - to)
        memory_usage.append(torch.cuda.memory_allocated(device) / 1024 / 1024)
    return inference_time, memory_usage


def benchmark_casent(
        entity_mentions: List[str],
        beam_size=None
):
    predictor = T5ForEntityTypingPredictor.from_pretrained('checkpoints/exp3_2')
    predictor.to(torch.device('cuda:0'))
    inference_time = []
    memory_usage = []
    for ent in entity_mentions:
        to = time.time()
        preds = predictor.predict_raw([ent], num_beams=beam_size, num_decode=beam_size)
        inference_time.append(time.time() - to)
        memory_usage.append(torch.cuda.memory_allocated(torch.device('cuda:0')) / 1024 / 1024)
    return inference_time, memory_usage


def benchmark_bert_base(
        entity_mentions: List[str]
):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10331)
    device = torch.device('cuda:0')
    model.to(torch.device('cuda:0'))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inference_time = []
    memory_usage = []
    for mention in entity_mentions:
        to = time.time()
        entity = mention[mention.index('<M>') + 3:mention.index('</M>')]
        inputs = tokenizer(
            entity, mention, return_tensors='pt', max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = outputs.logits > torch.tensor(0.5, device=device)
        inference_time.append(time.time() - to)
        memory_usage.append(torch.cuda.memory_allocated(device) / 1024 / 1024)
    return inference_time, memory_usage


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    examples = list(UFETDataset('test', 'data/ufet'))
    examples = random.sample(examples, 100)

    type_vocab = UFETDataset.get_type_vocab('data/ufet')
    entity_mentions = [ex.context for ex in examples]

    ts, ms = benchmark_bert_base(entity_mentions)
    print(f'BERT inference time: {np.mean(ts):.4f} +- {np.std(ts):.4f}')
    print(f'BERT memory usage: {np.mean(ms):.4f} +- {np.std(ms):.4f} Max: {max(ms):.4f}')

    ts, ms = benchmark_casent(entity_mentions)
    print(f'CASENT inference time: {np.mean(ts):.4f} +- {np.std(ts):.4f}')
    print(f'CASENT memory usage: {np.mean(ms):.4f} +- {np.std(ms):.4f} Max: {max(ms):.4f}')

    beamsize = 8
    ts, ms = benchmark_casent(entity_mentions, beam_size=beamsize)
    print(f'CASENT inference time (beam size {beamsize}): {np.mean(ts):.4f} +- {np.std(ts):.4f}')
    print(f'CASENT memory usage (beam size {beamsize}): {np.mean(ms):.4f} +- {np.std(ms):.4f} Max: {max(ms):.4f}')

    ts, ms = benchmark_lite(entity_mentions, type_vocab)
    print(f'LITE inference time: {np.mean(ts):.4f} +- {np.std(ts):.4f}')
    print(f'LITE memory usage: {np.mean(ms):.4f} +- {np.std(ms):.4f} Max: {max(ms):.4f}')


if __name__ == "__main__":
    main()
