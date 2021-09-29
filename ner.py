from typing import TypeVar

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
