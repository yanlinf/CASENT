import argparse
import os
import json
import stanza
from tqdm import tqdm
from casent.utils.ner_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', default='data/ner_typing_data/')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    stanza_pipeline = stanza.Pipeline(
        lang="en", processors="tokenize,pos,constituency", use_gpu=True
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for ds in ('jnlpba', 'bc5cdr', 'mit_restaurant', 'mit_movie', 'wnut17'):
        for split in ('dev', 'test'):
            outputs = []
            ner_examples = load_ner_examples(ds, split=split)

            if args.debug:
                ner_examples = ner_examples[:10]

            entity_mentions, all_labels = prepare_typing_examples(ner_examples)
            for entity, label in zip(tqdm(entity_mentions), all_labels):
                constituent_type = classify_mention_span(entity, stanza_pipeline)
                outputs.append({
                    'entity': entity,
                    'label': label,
                    'constituent_type': constituent_type,
                })

            output_path = os.path.join(args.output_dir, f'{ds}-{split}.json')
            with open(output_path, 'w') as file:
                json.dump(outputs, file, indent=2)


if __name__ == "__main__":
    main()
