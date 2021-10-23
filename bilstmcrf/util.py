import collections
from typing import Tuple, List, Dict

import torch
import torch.nn.utils.rnn as rnn

from data import PAD, UNK


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_epoch_time(start_time, end_time) -> Tuple[int, int]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def pad_batch(batch: Tuple[torch.LongTensor, torch.LongTensor]) \
    -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (xs, ys) = zip(*batch)
    x_lens = torch.LongTensor([len(x) for x in xs])
    x_pad = rnn.pad_sequence(xs, padding_value=0, batch_first=True)
    x_pad = x_pad.to(device)
    y_pad = rnn.pad_sequence(ys, padding_value=0, batch_first=True)
    y_pad = y_pad.to(device)
    return x_pad, x_lens, y_pad

def pad_test_batch(batch: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_lens = torch.LongTensor([len(x) for x in batch])
    x_pad= rnn.pad_sequence([x for x in batch], padding_value=0, batch_first=True)
    x_pad = x_pad.to(device)
    return x_pad, x_lens

def build_mappings(examples: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = set()
    for example in examples:
        for token in example:
            vocab.add(token)
    tokens_to_idx = collections.defaultdict(int)
    idx_to_tokens = collections.defaultdict(str)
    tokens_to_idx[PAD] = 0
    idx_to_tokens[0] = PAD
    tokens_to_idx[UNK] = 1
    idx_to_tokens[1] = UNK
    for i, token in enumerate(sorted(vocab)):
        tokens_to_idx[token] = i + 1
        idx_to_tokens[i + 1] = token
    return tokens_to_idx, idx_to_tokens

def format_output_labels(tags):
    pass

def entity_level_mean_f1(preds, gold):
    pass