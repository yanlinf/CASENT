from typing import List, Tuple, Optional, Any, Dict, Union
import stanza
import requests
import random
from collections import defaultdict
from dataclasses import dataclass
from datasets import load_dataset

HG_DATASETS_NAME = {
    'jnlpba': 'tner/bionlp2004',
    'bc5cdr': 'tner/bc5cdr',
    'mit_restaurant': 'tner/mit_restaurant',
    'mit_movie': 'tner/mit_movie_trivia',
    'wnut17': 'tner/wnut2017',
}

TYPE_MAPPING = {
    'jnlpba': {
        'DNA': 'DNA',
        'RNA': 'RNA',
        'cell_line': 'cell line',
        'cell_type': 'cell',
        'protein': 'protein',
    },
    'bc5cdr': {
        'Disease': 'disease',
        'Chemical': 'chemical',
    },
    'mit_restaurant': {
        'Rating': 'rating',
        'Amenity': 'amenity',
        'Location': 'location',
        'Restaurant_Name': 'restaurant',
        'Price': 'price',
        'Hours': 'hours',
        'Dish': 'dish',
        'Cuisine': 'cuisine',
    },
    'mit_movie': {
        'Actor': 'actor',
        'Plot': 'plot',
        'Opinion': 'opinion',
        'Award': 'award',
        'Year': 'year',
        'Genre': 'genre',
        'Origin': 'origin',
        'Director': 'director',
        'Soundtrack': 'soundtrack',
        'Relationship': 'relationship',
        'Character_Name': 'character',
        'Quote': 'quote'
    },
    'wnut17': {
        'corporation': 'corporation',
        'creative-work': 'creative work',
        'group': 'group',
        'location': 'location',
        'person': 'person',
        'product': 'product'
    },
    'tweetner7': {
        'corporation': 'corporation',
        'creative_work': 'creative work',
        'group': 'group',
        'location': 'location',
        'person': 'person',
        'product': 'product',
        'event': 'event'
    }
}


@dataclass
class NERExample:
    tokens: List[str]
    ner_tags: List[str]


def get_entities(ner_example: NERExample) -> List[Tuple[int, int, str]]:
    entities = []
    i = 0
    while i < len(ner_example.tokens):
        if ner_example.ner_tags[i].startswith('B-'):
            tag = ner_example.ner_tags[i][2:]
            j = i + 1
            while j < len(ner_example.tokens) and ner_example.ner_tags[j] == f'I-{tag}':
                j += 1
            entities.append((i, j, tag))
            i = j
        else:
            i += 1
    return entities


def load_ner_examples(dataset_name: str, split: str) -> List[NERExample]:
    hg_ds_name = HG_DATASETS_NAME[dataset_name]
    split_name = 'validation' if split == 'dev' else split
    eval_ds = load_dataset(hg_ds_name)[split_name]

    if hg_ds_name.startswith('tner/'):
        ner_examples = []
        url = f'https://huggingface.co/datasets/{hg_ds_name}/raw/main/dataset/label.json'
        label2id = requests.get(url).json()
        id2label = {i: t for t, i in label2id.items()}
        for example in eval_ds:
            ner_examples.append(NERExample(
                tokens=example['tokens'],
                ner_tags=[id2label[i] for i in example['tags']]
            ))

    else:
        ner_tagset = eval_ds.info.features['ner_tags'].feature.names

        ner_examples = []
        for example in eval_ds:
            ner_examples.append(NERExample(
                tokens=example['tokens'],
                ner_tags=[ner_tagset[k] for k in example['ner_tags']]
            ))
    return ner_examples


def prepare_typing_examples(
        ner_examples: List[NERExample],
) -> Tuple[List[str], List[str]]:
    entity_mentions = []
    all_labels = []
    for ex in ner_examples:
        for i, j, tag in get_entities(ex):
            mention = ' '.join(ex.tokens[:i] + ['<M>'] + ex.tokens[i:j] + ['</M>'] + ex.tokens[j:])
            all_labels.append(tag)
            entity_mentions.append(mention)
    return entity_mentions, all_labels


def extract_constituent_spans(
        root: stanza.models.constituency.parse_tree.Tree
) -> List[Tuple[int, int, str]]:
    """
    generate a list of (start_token_idx, end_token_idx) that represents
    the constituents using depth-first-search.
    """

    def dfs(curr_node):
        nonlocal start_token
        if curr_node.is_leaf():
            start_token += 1
        else:
            res.append((
                start_token,
                start_token + len(curr_node.leaf_labels()),
                curr_node.label
            ))
            for child in curr_node.children:
                dfs(child)

    start_token = 0
    res = []
    dfs(root)
    return res


def classify_mention_span(
        entity_mention: str,
        stanza_pipeline,
) -> Optional[str]:
    mention = entity_mention[entity_mention.index('<M>') + 3:entity_mention.index('</M>')].strip()

    text = entity_mention.replace('<M>', '').replace('</M>', '')
    doc = stanza_pipeline(text)

    for i, sentence in enumerate(doc.sentences):
        for start_token, end_token, label in extract_constituent_spans(sentence.constituency):
            start_char = sentence.tokens[start_token].start_char
            end_char = sentence.tokens[end_token - 1].end_char
            span = text[start_char:end_char].strip()
            if span == mention:
                return label
    return None


def subsample_by_class(
        samples: List[Any],
        labels: List[Any],
        n_sample_per_class: Union[int, Dict[Any, int]],
        seed: Optional[int] = None
) -> Tuple[List[Any], List[Any]]:
    if seed is not None:
        random.seed(seed)

    class_to_samples = defaultdict(list)
    for sample, label in zip(samples, labels):
        class_to_samples[label].append(sample)

    subsampled_samples = []
    subsampled_labels = []
    for label, class_samples in class_to_samples.items():
        num = n_sample_per_class[label] if isinstance(n_sample_per_class, dict) else n_sample_per_class
        if len(class_samples) >= n_sample_per_class:
            chosen_samples = random.sample(class_samples, num)
        else:
            raise ValueError(f'Not enough samples for class {label}')
            # chosen_samples = class_samples

        subsampled_samples.extend(chosen_samples)
        subsampled_labels.extend([label] * len(chosen_samples))

    return subsampled_samples, subsampled_labels


def subsample_by_size(
        samples: List[Any],
        labels: List[Any],
        size: int,
        seed: Optional[int] = None,
        balanced: bool = False,
) -> Tuple[List[Any], List[Any]]:
    if seed is not None:
        random.seed(seed)

    ratio = size / len(samples)

    classes = sorted(set(labels), key=lambda x: -labels.count(x))
    subsamples, sublabels = [], []
    for i, c in enumerate(classes):
        class_samples = [samples[i] for i in range(len(samples)) if labels[i] == c]
        if i == len(classes) - 1:
            class_num = size - len(subsamples)
        else:
            if balanced:
                class_num = int(size / len(classes))
            else:
                class_num = int(len(class_samples) * ratio)

        # assert 0 < class_num <= len(class_samples)
        class_idx = random.sample(range(len(class_samples)), class_num)
        subsamples.extend([class_samples[i] for i in class_idx])
        sublabels.extend([c] * class_num)
    assert len(subsamples) == len(sublabels) == size
    return subsamples, sublabels


def subsample_by_ratio(
        samples: List[Any],
        labels: List[Any],
        ratio: float,
        seed: Optional[int] = None,
        balanced: bool = False,
) -> Tuple[List[Any], List[Any]]:
    return subsample_by_size(samples, labels, int(len(samples) * ratio), seed, balanced)
