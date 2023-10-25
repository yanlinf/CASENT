from transformers import AutoTokenizer
import os
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EntityTypingOutput:
    types: List[str]
    scores: List[float]


class AutoTokenizerForEntityTyping:
    special_tokens_to_add = ['<M>', '</M>']

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if not (os.path.isdir(pretrained_model_name_or_path)
                or pretrained_model_name_or_path.startswith('yanlinf/casent')):
            assert all(t not in tokenizer.vocab for t in cls.special_tokens_to_add)
            tokenizer.add_special_tokens(
                {'additional_special_tokens': tokenizer.additional_special_tokens + cls.special_tokens_to_add})
        else:
            assert all(t in tokenizer.vocab for t in cls.special_tokens_to_add)
        return tokenizer
