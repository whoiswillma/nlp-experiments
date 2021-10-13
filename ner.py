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


def extract_named_entity_spans_from_bio(tags: list[str]) -> dict[str, list[tuple[int,int]]]:
    """Convert BIO tags to named entity spans (inclusive) by type
    """

    named_entity_spans: dict[str, list[tuple[int, int]]] = {}

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
