import random
import argparse
from tqdm import tqdm
from casent.dataset import *
from casent.utils.llm_backend import *
from casent.utils.ufet_utils import ufet_f1


def format_ufet(example: EntityTypingExample,
                is_test: bool = False):
    entity = example.context.replace('<M> ', '<mark>').replace(' </M>', '</mark>')
    res = f'Entity: {entity}\nLabels:'
    if not is_test:
        res += ", ".join(example.labels)
    return res


def process_response(resp: str):
    return [t.strip().replace('_', ' ').lower() for t in resp.split(',')]


def run_ufet_chatgpt(
        train_examples: List[EntityTypingExample],
        test_examples: List[EntityTypingExample],
        type_vocab: List[str],
        n_shot: int = 32,
        seed: int = 0
) -> List[Dict]:
    random.seed(seed)
    demos = random.sample(train_examples, n_shot)
    demos = [format_ufet(demo, is_test=False) for demo in demos]
    instruction = 'Instruction: Predict the fine-grained entity types for the entity mention tagged by <mark>. Separate the types with commas.'
    res = []
    type_vocab = set(type_vocab)
    for i, ex in enumerate(tqdm(test_examples)):
        target = format_ufet(ex, is_test=True)
        prompt = '\n\n'.join([instruction] + demos + [target])
        if i == 0:
            print('ChatGPT prompt:')
            print(prompt)
        response = few_shot_query(
            prompt=prompt,
            engine='gpt-3.5-turbo-0301',
            max_tokens=128,
            stop_token='\n',
            temperature=0.0,
            top_p=1.0,
        )
        preds = process_response(response)
        res.append({
            'entity': ex.context,
            'prompt': prompt,
            'labels': ex.labels,
            'raw_response': response,
            'raw_predictions': preds,
            'predictions': [t for t in preds if t in type_vocab],
            'scores': [0.0] * len(preds),
        })
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', default='checkpoints/chatgpt_ufet/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_shot', default=32, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    print(args)
    print()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_examples = list(UFETDataset('train', 'data/ufet/'))
    test_examples = list(UFETDataset('test', 'data/ufet/'))

    if args.debug:
        test_examples = test_examples[:10]

    type_vocab = UFETDataset.get_type_vocab('data/ufet/')
    res = run_ufet_chatgpt(
        train_examples,
        test_examples,
        type_vocab,
        n_shot=args.n_shot,
        seed=args.seed,
    )
    all_labels = [ex.labels for ex in test_examples]
    all_preds = [d['predictions'] for d in res]
    p, r, f1 = ufet_f1(all_preds, all_labels)
    print(f'P: {p:.4f}, R: {r:.4f} F1: {f1:.4f}')

    output_path = os.path.join(
        args.output_dir,
        f'ufet-test-{args.n_shot}shot-seed{args.seed}-results.json'
    )

    with open(output_path, 'w') as file:
        json.dump(res, file, indent=2)

    print(f'Saved results to {output_path}')


if __name__ == "__main__":
    main()
