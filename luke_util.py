import logging
import random
from typing import Collection, Optional, Union

import torch
from transformers import LukeConfig, LukeForEntitySpanClassification, LukeModel, LukeTokenizer


def chunked(collection: Collection, n: int) -> Collection:
    l = len(collection)
    return [collection[i: min(i + n, l)] for i in range(0, l, n)]


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


def take_closure_over_entity_spans_to_labels(
        entity_spans_to_labels: dict[tuple[int, int], int]
) -> dict[tuple[int, int], int]:
    """Returns a closure over `entity_spans_to_labels` such that, for every
    entity span (i, j) and associated label l, all sub spans (i', j') such that
    i <= i' < j' <= j is associated with l in its closure.
    """

    if not entity_spans_to_labels:
        return {}

    idx_to_label: list[Optional[int]] = [None] * max(end for _, end in entity_spans_to_labels)
    for entity_span, label in entity_spans_to_labels.items():
        start, end = entity_span
        for i in range(start, end):
            idx_to_label[i] = label

    closure: dict[tuple[int, int], int] = {}

    def add_all_spans_to_closure(start, end, label):
        for inner_start in range(start, end):
            for inner_end in range(inner_start + 1, end + 1):
                closure[(inner_start, inner_end)] = label

    prev_label = None
    start_index = None

    # causes loop body to execute one more time if current_label != None
    idx_to_label += [None]

    for i, label in enumerate(idx_to_label):
        if prev_label != label:
            if prev_label:
                add_all_spans_to_closure(start_index, i, prev_label)

            prev_label = label
            start_index = i

    return closure


def take_closure_over_entity_spans(
        entity_spans: Collection[tuple[int, int]]
) -> Collection[tuple[int, int]]:
    fake_entity_spans_to_labels = { entity_span: 0 for entity_span in entity_spans }
    fake_entity_spans_to_labels = take_closure_over_entity_spans_to_labels(fake_entity_spans_to_labels)
    return fake_entity_spans_to_labels.keys()


def get_entity_char_spans_and_labels(
        tokens: list[str],
        entity_spans_to_labels: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], list[int]]:

    starts, ends = get_word_start_end_positions(tokens)
    entity_spans_to_labels = take_closure_over_entity_spans_to_labels(entity_spans_to_labels)

    entity_char_spans: list[tuple[int, int]] = []
    labels: list[int] = []

    for (start_token_idx, end_token_idx), label in entity_spans_to_labels.items():
        entity_char_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))
        labels.append(label)

    return entity_char_spans, labels


def get_nonentity_char_spans(
        tokens: list[str],
        entity_spans: Collection[tuple[int, int]],
        max_span_len: Optional[int] = None,
        choose_k: Optional[int] = None
) -> list[tuple[int, int]]:

    entity_spans = take_closure_over_entity_spans(entity_spans)
    num_tokens = len(tokens)
    starts, ends = get_word_start_end_positions(tokens)

    max_span_len = max_span_len or num_tokens

    non_entity_char_spans = []
    for start_token_idx in range(0, num_tokens):
        for end_token_idx in range(start_token_idx + 1, min(num_tokens, start_token_idx + max_span_len) + 1):
            if (start_token_idx, end_token_idx) not in entity_spans:
                non_entity_char_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))

    if choose_k is not None:
        assert choose_k > 0
        non_entity_char_spans = random.choices(non_entity_char_spans, k=choose_k)

    return non_entity_char_spans


def get_entity_and_nonentity_char_spans_and_labels(
        tokens: list[str],
        entity_spans_to_labels: dict[tuple[int, int], int],
        nonentity_label: int,
        max_nonentity_span_len: Optional[int] = 16,
        nonentity_choose_k: Optional[int] = None
) -> tuple[list[tuple[int, int]], list[int]]:

    entity_char_spans, labels = get_entity_char_spans_and_labels(
        tokens,
        entity_spans_to_labels
    )
    nonentity_char_spans = get_nonentity_char_spans(
        tokens,
        entity_spans_to_labels.keys(),
        max_span_len=max_nonentity_span_len,
        choose_k=nonentity_choose_k
    )
    labels += [nonentity_label] * len(nonentity_char_spans)
    char_spans = entity_char_spans + nonentity_char_spans
    assert len(labels) == len(char_spans)

    return char_spans, labels


def make_model_and_tokenizer(num_labels):
    # make luke model and tokenizer
    logging.info('Initializing Model and Tokenizer')
    config = LukeConfig() 
    config.num_labels = num_labels

    model = LukeForEntitySpanClassification(config)
    model.luke = LukeModel.from_pretrained('studio-ousia/luke-base')
    tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-base', task='entity_span_classification')
    logging.info('Model initialized fresh')
    logging.info(f'config = {config}')
    logging.info(f'model = {model}')
    logging.info(f'tokenizer = {tokenizer}')

    return model, tokenizer


def make_train_stats_dict():
    return {
        'loss': 0.0,
        'num_spans': 0
    }


def train_luke_model(
        model,
        tokenizer,
        tokens: list[str],
        entity_spans_to_labels: dict[tuple[int, int], int],
        nonentity_label: int,
        stats: dict[str, any],
        nonentity_choose_k: Union[int, str] = 'all'
):

    model.train()

    if nonentity_choose_k == 'num_entity_spans':
        nonentity_choose_k = len(entity_spans_to_labels)
    elif nonentity_choose_k == 'all':
        nonentity_choose_k = None

    assert nonentity_choose_k is None or type(nonentity_choose_k) is int

    all_char_spans, labels = get_entity_and_nonentity_char_spans_and_labels(
        tokens,
        entity_spans_to_labels,
        nonentity_label,
        nonentity_choose_k=nonentity_choose_k
    )
    text = ' '.join(tokens)

    inputs = tokenizer(
        text,
        entity_spans=all_char_spans,
        return_tensors='pt',
        return_length=True
    )

    # TODO: how to determine max length from luke model / tokenizer?
    if inputs.length > 512:
        raise ValueError(f'Input is too long: inputs.length={inputs.length}')
    del inputs['length']

    outputs = model(**inputs, labels=torch.tensor(labels).unsqueeze(0))
    outputs.loss.backward()

    stats['loss'] += outputs.loss.item()
    stats['num_spans'] += len(labels)


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
        raise ValueError(f'Expected entity_span_level to be either "token" or "char". '
                         f'Got {entity_span_level} instead.')

    text = ' '.join(tokens)

    inputs = tokenizer(
        text,
        entity_spans=entity_char_spans,
        return_tensors='pt',
        return_length = True
    )

    # TODO: how to determine max length from luke model / tokenizer?
    if inputs.length > 512:
        raise ValueError(f'Input is too long: inputs.length={inputs.length}')
    del inputs['length']

    outputs = model(**inputs)
    result = outputs.logits.argmax(-1).squeeze().tolist()

    if isinstance(result, list):
        return result
    else:
        # if the tensor contains a single element, tolist() returns a scalar
        assert isinstance(result, int)
        return [result]


def acid_test_luke_model(
        model,
        tokenizer,
        tokens: list[str],
        entity_spans_to_labels: dict[tuple[int, int], int],
        nonentity_label: int
):
    all_char_spans, labels = get_entity_and_nonentity_char_spans_and_labels(
        tokens,
        entity_spans_to_labels,
        nonentity_label,
        nonentity_choose_k=max(len(tokens) // 2, len(entity_spans_to_labels))
    )

    assert len(all_char_spans) == len(labels)

    predictions = test_luke_model_on_entity_spans(
        model,
        tokenizer,
        tokens,
        entity_spans=all_char_spans,
        entity_span_level='char'
    )

    # logging.debug(f'labels = {labels}, predictions = {predictions}')
    assert len(labels) == len(predictions)

    correct = sum([1 for pred, label in zip(labels, predictions) if pred == label])
    total = len(labels)

    return correct, total

