import logging
import random
from typing import Collection, Optional, Union, TypeVar

import torch
from transformers import (
    LukeConfig,
    LukeForEntitySpanClassification,
    LukeModel,
    LukeTokenizer,
)

import util
from ner import NamedEntityIdSpans

# an entity token span is a token-level span including the LHS, *excluding* the
# RHS
EntityTokenSpan = tuple[int, int]


# an entity char span is a char-level span including the LHS, *excluding* the
# RHS
EntityCharSpan = tuple[int, int]


# either a token span or char span, depending on context
EntitySpan = Union[EntityTokenSpan, EntityCharSpan]


T = TypeVar("T")


def chunked(collection: T, n: int) -> list[T]:
    l = len(collection)
    return [collection[i : min(i + n, l)] for i in range(0, l, n)]


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


def map_to_char_span(
    start_end_positions: tuple[list[int], list[int]], token_spans: list[EntityTokenSpan]
) -> list[EntityCharSpan]:
    start, end = start_end_positions
    # we minus one from x[1] because x[1] is exclusive, so we want the end pos
    # of the token *before* x[1]
    return list(map(lambda x: (start[x[0]], end[x[1] - 1]), token_spans))


def take_closure_over_entity_spans_to_labels(
    entity_spans_to_labels: dict[EntityTokenSpan, int]
) -> dict[EntityTokenSpan, int]:
    """Returns a closure over `entity_spans_to_labels` such that, for every
    entity span (i, j) and associated label l, all sub spans (i', j') such that
    i <= i' < j' <= j is associated with l in its closure.
    """

    if len(entity_spans_to_labels) == 0:
        return {}

    idx_to_label: list[Optional[int]] = [None] * max(
        end for _, end in entity_spans_to_labels
    )
    for entity_span, label in entity_spans_to_labels.items():
        start, end = entity_span
        for i in range(start, end):
            idx_to_label[i] = label

    closure: dict[EntityTokenSpan, int] = {}

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
            if prev_label is not None:
                add_all_spans_to_closure(start_index, i, prev_label)

            prev_label = label
            start_index = i

    return closure


def take_closure_over_entity_spans(
    entity_spans: Collection[EntityTokenSpan],
) -> set[EntityTokenSpan]:
    fake_entity_spans_to_labels = {entity_span: 0 for entity_span in entity_spans}
    fake_entity_spans_to_labels = take_closure_over_entity_spans_to_labels(
        fake_entity_spans_to_labels
    )
    return set(fake_entity_spans_to_labels.keys())


def get_entity_char_spans_and_labels(
    tokens: list[str], entity_spans_to_labels: dict[EntityTokenSpan, int],
) -> tuple[list[EntityCharSpan], list[int]]:

    starts, ends = get_word_start_end_positions(tokens)

    entity_char_spans: list[tuple[int, int]] = []
    labels: list[int] = []

    for (start_token_idx, end_token_idx), label in entity_spans_to_labels.items():
        entity_char_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))
        labels.append(label)

    return entity_char_spans, labels


def list_all_spans(
    num_tokens: int,
    max_span_len: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Union[list[EntityTokenSpan], list[list[EntityTokenSpan]]]:

    result = []

    for i in range(num_tokens):
        for j in range(i + 1, min(num_tokens + 1, i + max_span_len + 1)):
            result.append((i, j))

    if batch_size is not None:
        return chunked(result, batch_size)
    else:
        return result


def get_nonentity_char_spans(
    tokens: list[str],
    entity_spans: Collection[EntityTokenSpan],
    max_span_len: Optional[int] = None,
    choose_k: Optional[int] = None,
) -> list[EntityCharSpan]:

    num_tokens = len(tokens)
    starts, ends = get_word_start_end_positions(tokens)

    max_span_len = max_span_len or num_tokens

    non_entity_char_spans = []
    for start_token_idx, end_token_idx in list_all_spans(num_tokens, max_span_len):
        if (start_token_idx, end_token_idx) not in entity_spans:
            non_entity_char_spans.append(
                (starts[start_token_idx], ends[end_token_idx - 1])
            )

    if choose_k is not None:
        assert choose_k > 0
        non_entity_char_spans = random.choices(non_entity_char_spans, k=choose_k)

    return non_entity_char_spans


def get_entity_and_nonentity_char_spans_and_labels(
    tokens: list[str],
    entity_spans_to_labels: dict[EntityTokenSpan, int],
    nonentity_label: int,
    max_nonentity_span_len: Optional[int] = 16,
    nonentity_choose_k: Optional[int] = None,
) -> tuple[list[EntityCharSpan], list[int]]:

    entity_char_spans, labels = get_entity_char_spans_and_labels(
        tokens, entity_spans_to_labels
    )
    nonentity_char_spans = get_nonentity_char_spans(
        tokens,
        entity_spans_to_labels.keys(),
        max_span_len=max_nonentity_span_len,
        choose_k=nonentity_choose_k,
    )
    labels += [nonentity_label] * len(nonentity_char_spans)
    char_spans = entity_char_spans + nonentity_char_spans
    assert len(labels) == len(char_spans)

    return char_spans, labels


def make_model_and_tokenizer(num_labels):
    # make luke model and tokenizer
    logging.info("Initializing Model and Tokenizer")
    config = LukeConfig()
    config.num_labels = num_labels

    model = LukeForEntitySpanClassification(config)
    model.luke = LukeModel.from_pretrained("studio-ousia/luke-base")
    tokenizer = LukeTokenizer.from_pretrained(
        "studio-ousia/luke-base", task="entity_span_classification"
    )
    logging.info("Model initialized fresh")
    logging.info(f"config = {config}")
    logging.info(f"model = {model}")
    logging.info(f"tokenizer = {tokenizer}")

    model = model.to(util.PTPU)

    return model, tokenizer


def make_train_stats_dict():
    return {"loss": 0.0, "num_spans": 0}


def train_luke_model(
    model,
    tokenizer,
    tokens: list[str],
    entity_spans_to_labels: dict[EntityTokenSpan, int],
    nonentity_label: int,
    stats: dict[str, any],
    nonentity_choose_k: Union[int, str] = "all",
    example_id: Optional[int] = None,
):

    model.train()

    if nonentity_choose_k == "num_entity_spans":
        nonentity_choose_k = len(entity_spans_to_labels)
    elif nonentity_choose_k == "all":
        nonentity_choose_k = None

    assert nonentity_choose_k is None or type(nonentity_choose_k) is int

    all_char_spans, labels = get_entity_and_nonentity_char_spans_and_labels(
        tokens,
        entity_spans_to_labels,
        nonentity_label,
        nonentity_choose_k=nonentity_choose_k,
    )
    text = " ".join(tokens)

    spans_per_batch = len(all_char_spans)
    num_spans_trained = 0

    while num_spans_trained != len(all_char_spans):
        assert num_spans_trained < len(all_char_spans)

        if spans_per_batch == 0:
            raise RuntimeError(
                f"Unable to train on example {example_id} due to " f"memory issues"
            )

        start_idx = num_spans_trained
        end_idx = min(len(all_char_spans), start_idx + spans_per_batch)

        char_spans_to_train = all_char_spans[start_idx:end_idx]
        labels_to_train = labels[start_idx:end_idx]

        try:
            inputs = tokenizer(
                text,
                entity_spans=char_spans_to_train,
                return_tensors="pt",
                return_length=True,
            ).to(util.PTPU)

            labels_to_train = torch.tensor(labels_to_train).unsqueeze(0).to(util.PTPU)

        except RuntimeError as e:
            util.free_memory()

            logging.warning(f"Example {example_id}: {e}")
            spans_per_batch //= 2
            logging.warning(
                f"Example {example_id}: Halving number of spans per "
                f"train iter to {spans_per_batch} and trying again"
            )
            continue

        # TODO: how to determine max length from luke model / tokenizer?
        if inputs.length > 512:
            util.free_memory()

            logging.warning(f"Input is too long: inputs.length={inputs.length}")
            spans_per_batch //= 2
            logging.warning(
                f"Example {example_id}: Halving number of spans per "
                f"train iter to {spans_per_batch} and trying again"
            )
            continue

        del inputs["length"]

        try:
            outputs = model(**inputs, labels=labels_to_train)
            outputs.loss.backward()

            stats["loss"] += outputs.loss.item()
            stats["num_spans"] += labels_to_train.shape[1]
            num_spans_trained += end_idx - start_idx

        except RuntimeError as e:
            util.free_memory()

            logging.warning("Could not backprop model")
            spans_per_batch //= 2
            logging.warning(
                f"Example {example_id}: Halving number of spans per "
                f"train iter to {spans_per_batch} and trying again"
            )
            continue


def test_luke_model_on_entity_spans(
    model,
    tokenizer,
    tokens: list[str],
    entity_spans: list[EntitySpan],
    entity_span_level: str,
) -> list[int]:

    model.eval()

    if entity_span_level == "token":
        starts, ends = get_word_start_end_positions(tokens)

        entity_char_spans = []
        for start_token_idx, end_token_idx in entity_spans:
            entity_char_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))

    elif entity_span_level == "char":
        entity_char_spans = entity_spans

    else:
        raise ValueError(
            f'Expected entity_span_level to be either "token" or "char". '
            f"Got {entity_span_level} instead."
        )

    text = " ".join(tokens)

    inputs = tokenizer(
        text, entity_spans=entity_char_spans, return_tensors="pt", return_length=True
    ).to(util.PTPU)

    # TODO: how to determine max length from luke model / tokenizer?
    if inputs.length > 512:
        raise ValueError(f"Input is too long: inputs.length={inputs.length}")
    del inputs["length"]

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
    entity_spans_to_labels: dict[EntityTokenSpan, int],
    nonentity_label: int,
):
    all_char_spans, labels = get_entity_and_nonentity_char_spans_and_labels(
        tokens,
        entity_spans_to_labels,
        nonentity_label,
        nonentity_choose_k=max(len(tokens) // 2, len(entity_spans_to_labels)),
    )

    assert len(all_char_spans) == len(labels)

    predictions = test_luke_model_on_entity_spans(
        model, tokenizer, tokens, entity_spans=all_char_spans, entity_span_level="char"
    )

    logging.debug(f"labels = {labels}, predictions = {predictions}")
    assert len(labels) == len(predictions)

    correct = sum([1 for pred, label in zip(labels, predictions) if pred == label])
    total = len(labels)

    return correct, total


def convert_span_labels_to_named_entity_spans(
    span_labels: set[tuple[EntityTokenSpan, int]]
) -> NamedEntityIdSpans:
    named_entity_spans: NamedEntityIdSpans = {}

    for span, label in span_labels:
        if label not in named_entity_spans:
            named_entity_spans[label] = []

        start, end = span
        # need to subtract 1 from end since NamedEntityIdSpan is inclusive
        named_entity_spans[label].append((start, end - 1))

    for spans in named_entity_spans.values():
        spans.sort(key=lambda span: span[0])

    return named_entity_spans


def greedy_extract_named_entity_spans(
    span_label_logit: list[tuple[EntityTokenSpan, int, float]], nonentity_label: int
) -> NamedEntityIdSpans:
    """Implements greedy span selection algorithm from LUKE paper.

    p. 6
    > During the inference, we first exclude all spans classified into the non-
    > entity type. To avoid selecting overlapping spans, we greedily select a
    > span from the remaining spans based on the logit of its predicted entity
    > type in descending order if the span does not overlap with those already
    > selected.

    """

    # make a shallow copy
    span_label_logit = list(span_label_logit)

    # remove all spans of nonentity type.

    for i, (span, label, logit) in reversed(list(enumerate(span_label_logit))):
        if label == nonentity_label:
            del span_label_logit[i]

    # sort remaining spans by logit
    span_label_logit.sort(key=lambda x: x[2], reverse=True)

    selected_span_labels: set[tuple[EntityTokenSpan, int]] = set()

    def overlaps_with_selected(span: EntityTokenSpan) -> bool:
        for other, _ in selected_span_labels:
            if max(span[0], other[0]) < min(span[1], other[1]):
                return True
        return False

    for span, label, _ in span_label_logit:
        if not overlaps_with_selected(span):
            # check that we're not inserting any nonentity labels into our
            # selected spans
            assert label != nonentity_label

            selected_span_labels.add((span, label))

    return convert_span_labels_to_named_entity_spans(selected_span_labels)


def eval_named_entity_spans(
    model: LukeModel,
    tokenizer: LukeTokenizer,
    tokens: list[str],
    nonentity_label: int,
    max_span_len: Optional[int],
) -> NamedEntityIdSpans:

    model.eval()

    start_end_positions = get_word_start_end_positions(tokens)
    text = " ".join(tokens)
    token_spans: list[list[EntityTokenSpan]] = list_all_spans(
        len(tokens), max_span_len, 16
    )

    span_label_logit: list[tuple[EntityTokenSpan, int, float]] = []

    for batch in token_spans:
        entity_char_spans = map_to_char_span(start_end_positions, batch)

        inputs = tokenizer(
            text,
            entity_spans=entity_char_spans,
            return_tensors="pt",
            return_length=True,
        ).to(util.PTPU)

        # TODO: how to determine max length from luke model / tokenizer?
        if inputs.length > 512:
            raise ValueError(f"Input is too long: inputs.length={inputs.length}")
        del inputs["length"]

        outputs = model(**inputs)
        max_logits, argmax = torch.max(outputs.logits.squeeze(0), -1)
        for span, logit, label in zip(batch, max_logits, argmax):
            val = (span, label.item(), logit.item())
            assert val not in span_label_logit
            span_label_logit.append(val)

    return greedy_extract_named_entity_spans(span_label_logit, nonentity_label)
