from typing import List, Tuple

def extract_nets(tokens: List[str], tags: List[str]) -> List[Tuple[str, str]]:
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
