from typing import TypeVar, Optional


T = TypeVar('T')


def extract_nets_from_chunks(tokens: list[str], labels: list[T], null_label: T) -> list[tuple[str, T]]:
    named_entity_and_type: list[tuple[str, T]] = []

    current_named_entity = []
    current_label = null_label

    for token, label in list(zip(tokens, labels)) + [('', null_label)]:
        if current_label != label:
            if current_label != null_label:
                named_entity_and_type.append((' '.join(current_named_entity), current_label))
            current_named_entity = []

        current_label = label
        current_named_entity.append(token)

    return named_entity_and_type


def extract_nets_from_bio(tokens: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """Extract named entity and types (nets) from tokens, tags in BIO format
    """

    named_entities_and_category = []
    
    current_named_entity = []
    current_tag = None
    
    for token, tag_str in zip(tokens, tags):
        bio = tag_str[:1]
        tag = tag_str[2:]
        
        if bio == 'B':
            if current_tag:
                named_entities_and_category.append((' '.join(current_named_entity), current_tag))
            
            current_named_entity = [token]
            current_tag = tag
            
        elif bio == 'O':
            if current_tag:
                named_entities_and_category.append((' '.join(current_named_entity), current_tag))
            
            current_named_entity = []
            current_tag = None
            
        elif bio == 'I':
            if not (tag == current_tag):
                raise Exception(f'Invalid transition {current_tag} -> I-{tag}')
            current_named_entity.append(token)
            
        else:
            raise Exception(f'Unexpected tag string {tag_str}')
        
    return named_entities_and_category


# dict from label to token-level spans inclusive
NamedEntitySpans = dict[str, list[tuple[int, int]]]


def extract_named_entity_spans_from_bio(tags: list[str]) -> NamedEntitySpans:
    """Convert BIO tags to named entity spans (inclusive) by type
    """

    named_entity_spans: NamedEntitySpans = {}

    current_named_entity_start: Optional[int] = None
    current_tag: Optional[str] = None

    def add_current_named_entity_to_span(end_index: int):
        assert current_tag is not None
        assert current_named_entity_start is not None

        if current_tag not in named_entity_spans:
            named_entity_spans[current_tag] = []

        named_entity_spans[current_tag].append((current_named_entity_start, end_index - 1))

    for i, tag_str in enumerate(tags + ['O']):
        bio = tag_str[:1]
        tag = tag_str[2:]

        if bio == 'B':
            if current_tag is not None:
                add_current_named_entity_to_span(i)

            current_named_entity_start = i
            current_tag = tag

        elif bio == 'O':
            if current_tag is not None:
                add_current_named_entity_to_span(i)

            current_named_entity_start = None
            current_tag = None

        elif bio == 'I':
            if not (tag == current_tag):
                raise ValueError(f'Invalid transition {current_tag} -> I-{tag}')

        else:
            raise ValueError(f'Unexpected tag string {tag_str}')

    return named_entity_spans


class NERBinaryConfusionMatrix:

    def __init__(
            self,
            tp: int = 0,
            fn: int = 0,
            fp: int = 0
    ):
        # true positives (pred and gold both contain span of same type)
        self.tp = tp

        # false negatives (gold contains a span not matched in pred)
        self.fn = fn

        # false positives (pred contains a span not matched in gold)
        self.fp = fp

    def __eq__(self, other):
        if not isinstance(other, NERBinaryConfusionMatrix):
            return False

        return self.tp == other.tp and self.fn == other.fn and self.fp == other.fp

    def __repr__(self):
        return f'{{ tp: {self.tp}, fn: {self.fn}, fp: {self.fp} }}'


def compute_binary_confusion_matrix_from_bio(
        pred_bio: list[str],
        gold_bio: list[str],
        accumulate: Optional[NERBinaryConfusionMatrix] = None
) -> NERBinaryConfusionMatrix:
    assert len(pred_bio) == len(gold_bio)

    pred_ner_spans = extract_named_entity_spans_from_bio(pred_bio)
    gold_ner_spans = extract_named_entity_spans_from_bio(gold_bio)

    return compute_binary_confusion_from_named_entity_spans(
        pred_ner_spans,
        gold_ner_spans,
        accumulate=accumulate
    )


def compute_binary_confusion_from_named_entity_spans(
        pred_ne_spans: NamedEntitySpans,
        gold_ne_spans: NamedEntitySpans,
        accumulate: Optional[NERBinaryConfusionMatrix] = None
) -> NERBinaryConfusionMatrix:

    if accumulate is None:
        accumulate = NERBinaryConfusionMatrix()

    # update true positives, and edit pred_ner_spans and gold_ner_spans to
    # contain disjoint spans
    named_entity_types: set[str] = pred_ne_spans.keys() | gold_ne_spans.keys()
    for net in named_entity_types:
        if net in pred_ne_spans and net in gold_ne_spans:
            for span in list(pred_ne_spans[net]):
                if span in gold_ne_spans[net]:
                    accumulate.tp += 1
                    pred_ne_spans[net].remove(span)
                    gold_ne_spans[net].remove(span)

    # sanity check
    for net in named_entity_types:
        if net in pred_ne_spans and net in gold_ne_spans:
            for span in pred_ne_spans[net]:
                assert span not in gold_ne_spans[net]
            for span in gold_ne_spans[net]:
                assert span not in pred_ne_spans[net]

    accumulate.fn += sum(len(spans) for spans in gold_ne_spans.values())
    accumulate.fp += sum(len(spans) for spans in pred_ne_spans.values())

    return accumulate

