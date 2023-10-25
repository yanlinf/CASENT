import collections

from torch.utils.data import Dataset
from typing import List, Optional
from dataclasses import dataclass
import os
import json

DATASET_NUM_LABELS_MAPPING = {
    'ufet': 10331,
}


@dataclass
class EntityTypingExample:
    id: Optional[str]
    context: str
    labels: Optional[List[str]] = None


class UFETDataset(Dataset):
    """
    Ultra-fine entity typing dataset (Choi et al., 2018)

    train/dev/test splits are .jsonl files with each line in the following format:
        {
          'annot_id': 'APW_ENG_20101103.0611:12:2',
          'mention_span': 'Web users',
          'right_context_token': ['to', 'locate', 'its', ...],
          'y_str': ['citizen', 'person', 'user'],
          'left_context_token': ['The', 'British', 'Information', ...]
        }
    """

    def __init__(self, split, ufet_dir):
        self.split = split
        self.samples = self._load(split, ufet_dir)

    @classmethod
    def get_type_vocab(cls, ufet_dir):
        type_vocab_path = os.path.join(ufet_dir, 'ontology', 'types.txt')
        with open(type_vocab_path, 'r') as fin:
            type_vocab = [line.strip().replace('_', ' ') for line in fin]
        return type_vocab

    @classmethod
    def get_type_freq(cls, ufet_dir):
        type_freq = collections.defaultdict(int)
        train_set_path = os.path.join(ufet_dir, 'crowd', 'train.json')
        with open(train_set_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                dic = json.loads(line)
                for t in dic['y_str']:
                    type_freq[t.replace('_', ' ')] += 1
        return type_freq

    def _load(self, split, ufet_dir):
        path = os.path.join(ufet_dir, 'crowd', f'{split}.json')
        samples = []
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                dic = json.loads(line)
                labels = [t.replace('_', ' ') for t in dic['y_str']]
                if len(labels) == 0:
                    if split in ('dev', 'test'):
                        raise ValueError(f'{split} set contains a zero-label sample')
                    continue
                samples.append(
                    EntityTypingExample(
                        id=dic['annot_id'],
                        context=' '.join(dic['left_context_token']
                                         + ['<M>', dic['mention_span'], '</M>']
                                         + dic['right_context_token']),
                        labels=labels
                    )
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
