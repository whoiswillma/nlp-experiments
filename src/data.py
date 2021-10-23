import torch
import collections
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

UNK = '<UNK>'
PAD = '<P>'

class Conll2003(Dataset):
    def __init__(self, examples:List[str], labels:List[int],
                 idx_to_tokens:Dict[int, str], tokens_to_idx:Dict[str, int],
                 ner_tags:List[str], device: torch.device):
        self.examples = examples
        self.labels = labels
        self.ner_tags = ner_tags
        self.device = device

        # set up mappings
        self.tags_to_idx, self.idx_to_tags = self.process_tags(self.ner_tags)
        self.tokens_to_idx = tokens_to_idx
        self.idx_to_tokens = idx_to_tokens

        # map examples to encodings
        self.processed_examples = self.process_examples(self.examples)
        self.processed_labels = self.process_labels(self.labels)

    def process_tags(self, ner_tags: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        tags_to_idx = collections.defaultdict(int)
        idx_to_tags = collections.defaultdict(str)
        for i, tag in enumerate(ner_tags):
            tags_to_idx[tag] = i
            idx_to_tags[i] = tag
        return tags_to_idx, idx_to_tags

    def process_examples(self, examples: List[str]) -> List[torch.LongTensor]:
        proc_examples = []
        for example in examples:
            proc_example = torch.LongTensor([self.tokens_to_idx[t] for t in example]).to(self.device)
            proc_examples.append(proc_example)
        return proc_examples

    def process_labels(self, labels: List[int]) -> List[torch.LongTensor]:
        new_labels = []
        for label_vec in labels:
            new_labels.append(torch.LongTensor(label_vec))
        return new_labels

    def __getitem__(self, idx:int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.processed_examples[idx], self.processed_labels[idx]

    def __len__(self) -> int:
        return len(self.processed_examples)