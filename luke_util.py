import logging
import random
from typing import Collection, Optional
from transformers import LukeConfig, LukeForEntityClassification, LukeModel, LukeTokenizer
import torch

import util


def get_word_start_end_positions(tokens: list[str]) -> tuple[list[int], list[int]]:
    start_positions = []
    end_positions = []

    curr = 0

    for token in tokens:
        L = len(token)
        start_positions.append(curr)
        end_positions.append(curr + L)
        curr += L + 1

    return start_positions, end_positions


def get_entity_char_spans_and_labels(
        tokens: list[str],
        entity_spans_to_labels: dict[tuple[int, int], int],
):

    starts, ends = get_word_start_end_positions(tokens)

    entity_spans = []
    labels = []

    for (start_token_idx, end_token_idx), label in entity_spans_to_labels.items():
        entity_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))
        labels.append(label)

    return entity_spans, labels


def get_non_entity_char_spans(
        tokens: list[str],
        entity_spans: Collection[tuple[int, int]],
        max_span_len: Optional[int] = None,
        choose_k: Optional[int] = None
) -> list[tuple[int, int]]:

    num_tokens = len(tokens)
    starts, ends = get_word_start_end_positions(tokens)

    non_entity_spans = []
    for start_token_idx in range(0, num_tokens):
        for end_token_idx in range(start_token_idx + 1, num_tokens + 1):
            if max_span_len and end_token_idx - start_token_idx > max_span_len:
                continue

            if (start_token_idx, end_token_idx) not in entity_spans:
                non_entity_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))

    if choose_k:
        non_entity_spans = random.choices(non_entity_spans, k=choose_k)

    return non_entity_spans


def make_model_and_tokenizer(num_labels):
    # make luke model and tokenizer
    logging.info('Initializing Model and Tokenizer')
    config = LukeConfig() 
    config.num_labels = num_labels

    model = LukeForEntityClassification(config)
    model.luke = LukeModel.from_pretrained('studio-ousia/luke-base')
    tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-base', task='entity_classification')
    logging.info('Model initialized fresh')
    logging.info(f'config = {config}')
    logging.info(f'model = {model}')
    logging.info(f'tokenizer = {tokenizer}')

    return model, tokenizer


def make_train_stats_dict():
    return {
        'loss': 0.0,
        'num_backprops': 0
    }


def train_luke_model_on_entity_spans(
        model,
        tokenizer,
        tokens: list[str],
        entity_spans_to_labels: dict[tuple[int, int], int],
        stats: dict[str, any]
):
    model.train()

    entity_char_spans, labels = get_entity_char_spans_and_labels(tokens, entity_spans_to_labels)
    text = ' '.join(tokens)

    for entity_char_span, label in util.mytqdm(list(zip(entity_char_spans, labels)), desc='entities'):
        inputs = tokenizer(
            text,
            entity_spans=[entity_char_span],
            return_tensors='pt'
        )
        outputs = model(**inputs, labels=torch.tensor([label]))
        outputs.loss.backward()

        stats['loss'] += outputs.loss.item()
        stats['num_backprops'] += 1


def train_luke_model_on_non_entity_spans(
        model,
        tokenizer,
        tokens: list[str],
        entity_spans: Collection[tuple[int, int]],
        non_entity_label: int,
        stats: dict[str, any],
        choose_k: Optional[int] = None,
        max_span_len: Optional[int] = 15
):
    model.train()

    text = ' '.join(tokens)

    non_entity_char_spans = get_non_entity_char_spans(
        tokens,
        entity_spans,
        max_span_len=max_span_len,
        choose_k=choose_k
    )

    for non_entity_char_span in util.mytqdm(non_entity_char_spans, desc='non-entities'):
        inputs = tokenizer(
            text,
            entity_spans=[non_entity_char_span],
            return_tensors='pt'
        )
        outputs = model(**inputs, labels=torch.tensor([non_entity_label]))
        outputs.loss.backward()

        stats['loss'] += outputs.loss.item()
        stats['num_backprops'] += 1


def train_luke_model(
        model,
        tokenizer,
        tokens: list[str],
        entity_spans_to_labels: dict[tuple[int, int], int],
        non_entity_label: int,
        stats: dict[str, any]
):
    train_luke_model_on_entity_spans(
        model,
        tokenizer,
        tokens,
        entity_spans_to_labels,
        stats
    )

    train_luke_model_on_non_entity_spans(
        model,
        tokenizer,
        tokens,
        entity_spans_to_labels.keys(),
        non_entity_label,
        stats,
        choose_k=len(entity_spans_to_labels)
    )


def test_luke_model_on_entity_spans(
        model,
        tokenizer,
        tokens: list[str],
        entity_spans: list[tuple[int, int]],
        entity_span_level: str
) -> list[int]:
    model.eval()

    if entity_span_level == 'token':
        starts, ends = get_word_start_end_positions(tokens)

        entity_char_spans = []
        for start_token_idx, end_token_idx in entity_spans:
            entity_char_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))

    elif entity_span_level == 'char':
        entity_char_spans = entity_spans

    else:
        assert False

    text = ' '.join(tokens)
    labels = []
    for entity_char_span in util.mytqdm(entity_char_spans, desc='test'):
        inputs = tokenizer(
            text,
            entity_spans=[entity_char_span],
            return_tensors='pt'
        )
        outputs = model(**inputs)
        label = outputs.logits.argmax(-1).item()
        labels.append(label)

    return labels


def acid_test_luke_model(
        model,
        tokenizer,
        tokens: list[str],
        entity_spans_to_labels: dict[tuple[int, int], int],
        non_entity_label: int
):
    entity_char_spans, entity_labels = get_entity_char_spans_and_labels(
        tokens,
        entity_spans_to_labels
    )

    non_entity_char_spans = get_non_entity_char_spans(
        tokens,
        entity_spans_to_labels.keys(),
        choose_k=len(entity_char_spans)
    )

    char_spans = entity_char_spans + non_entity_char_spans

    labels = entity_labels + [non_entity_label] * len(non_entity_char_spans)
    predictions = test_luke_model_on_entity_spans(
        model,
        tokenizer,
        tokens,
        entity_spans=char_spans,
        entity_span_level='char'
    )
    assert len(labels) == len(predictions)

    logging.debug(f'labels = {labels}, predictions = {predictions}')

    correct = sum([1 for pred, label in zip(labels, predictions) if pred == label])
    total = len(labels)

    return correct, total

