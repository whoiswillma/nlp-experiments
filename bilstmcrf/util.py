import collections
from typing import Dict, List, Tuple

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

def build_token_mappings(examples: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
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
        tokens_to_idx[token] = i + 2
        idx_to_tokens[i + 2] = token
    return tokens_to_idx, idx_to_tokens

def build_tag_mappings(ner_tags: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    tags_to_idx = collections.defaultdict(int)
    idx_to_tags = collections.defaultdict(str)
    for i, tag in enumerate(ner_tags):
        tags_to_idx[tag] = i
        idx_to_tags[i] = tag
    return tags_to_idx, idx_to_tags

# metric code reference: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs/blob/master/validation.py
def compute_entity_level_f1(predicted_labels:List[List[str]], gold_labels:List[List[str]]) -> float:
    precision = compute_precision(predicted_labels=predicted_labels, gold_labels=gold_labels)
    recall = compute_precision(predicted_labels=gold_labels, gold_labels=predicted_labels)
    f1 = 0
    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def compute_precision(predicted_labels:List[List[str]], gold_labels:List[List[str]]) -> float:
    assert (len(predicted_labels) == len(gold_labels))
    num_correct = 0
    count = 0
    # loop through labels
    for labels_idx in range(len(predicted_labels)):
        pred = predicted_labels[labels_idx]
        gold = gold_labels[labels_idx]
        assert (len(pred) == len(gold))
        idx = 0
        while idx < len(pred):
            # start of a new entity span
            if pred[idx][0] == 'B':
                count += 1
                if pred[idx] == gold[idx]:
                    idx += 1
                    still_correct = True
                    # scan all I tags
                    while idx < len(pred) and pred[idx][0] == 'I':
                        if pred[idx] != gold[idx]:
                            still_correct = False
                        idx += 1
                    if idx < len(pred):
                        # the gold entity was longer than the pred
                        if gold[idx][0] == 'I':
                            still_correct = False
                    if still_correct:
                        num_correct += 1
                else:
                    idx += 1
            else:
                idx += 1
    precision = 0
    if count > 0:
        precision = float(num_correct) / count
    return precision