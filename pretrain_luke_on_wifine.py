import torch
from transformers import LukeTokenizer, LukeForEntityClassification, LukeConfig, LukeModel
import util
from wifineparse import wifine
import random
from tqdm import tqdm
import sys


torch.set_num_threads(1)


def flatten(ll: list[list]):
    return [e for l in ll for e in l]


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


def get_document_ids_to_train_with(num=None, fraction=None):
    if fraction: 
        num = fraction * len(wifine.DOCUMENT_INDEX.all_docids())
    
    return wifine.DOCUMENT_INDEX.all_docids()[:num]
    # return random.choices(wifine.DOCUMENT_INDEX.all_docids(), k=num)


def get_sentence_offsets(doc):
    """Returns a list that maps sent_idx -> token_idx
    """
    sentence_offsets = [0]
    for sent in doc.sentences_as_ids():
        sentence_offsets.append(sentence_offsets[-1] + len(sent))
    return sentence_offsets


def get_entity_token_span_to_figer_types(doc):
    sent_offs = get_sentence_offsets(doc)
    fe = doc.fine_entities()
    return {
        (sent_offs[s] + i, sent_offs[s] + j): t
        for i, j, t, s in zip(fe.begin, fe.end, fe.figer_types, fe.sent_idx)
    }


def get_entity_token_idx_to_figer_type(doc):
    sent_offs = get_sentence_offsets(doc)
    fe = doc.fine_entities()
    return {
        sent_offs[s] + i: t[0]
        for start, end, t, s in zip(fe.begin, fe.end, fe.figer_types, fe.sent_idx)
        for i in range(start, end)
    }


if __name__ == '__main__':
    # make luke model and tokenizer
    print('Initializing Model and Tokenizer')
    config = LukeConfig() 
    config.num_labels = len(wifine.FIGER_VOCAB) + 1
    
    # the label that reprents no entity type (i.e. 'O')
    NO_ENTITY_TYPE = len(wifine.FIGER_VOCAB)

    model = LukeForEntityClassification(config)
    model.luke = LukeModel.from_pretrained('studio-ousia/luke-base')
    tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-base', task='entity_classification')

    # get train dataset
    document_ids = get_document_ids_to_train_with(fraction=0.1)

    # learning setup
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    NUM_EPOCHS = 10

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Start epoch {epoch}')
        opt.zero_grad()
        total_loss = torch.tensor(0.0)

        # train one epoch
        for doc_id in document_ids:
            # extract resources
            document = wifine.DOCUMENT_INDEX.get_document(doc_id)

            text = document.sentences_as_tokens()
            text = flatten(text)
            num_tokens = len(text)
            starts, ends = get_word_start_end_positions(text)
            text = ' '.join(text)

            figer_types = get_entity_token_span_to_figer_types(document)

            # text: str
            # starts: token_idx -> char_idx
            # ends: token_idx -> char_idx
            # figer_types: (token_idx, token_idx) -> figer_type

            # train on figer types

            entity_spans = []
            labels = []
            for (start_token_idx, end_token_idx), figer_types in figer_types.items():
                labels.append(figer_types[0])
                entity_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))

            non_entity_spans = []
            for start_token_idx in range(num_tokens):
                for end_token_idx in range(num_tokens):
                    if (start_token_idx, end_token_idx) not in figer_types:
                        non_entity_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))
            non_entity_spans = random.choices(non_entity_spans, k=len(entity_spans))
            labels += [NO_ENTITY_TYPE] * len(non_entity_spans)

            assert len(labels) == len(entity_spans) + len(non_entity_spans)

            for entity_span, label in zip(entity_spans + non_entity_spans, labels):
                inputs = tokenizer(
                    text, 
                    entity_spans=[entity_span],
                    return_tensors='pt'
                )
                outputs = model(**inputs, labels=torch.tensor([label]))

                outputs.loss.backward()
                total_loss += outputs.loss

            opt.step()

        print(f'epoch {epoch} avg loss {total_loss / len(labels)}')

        num_correct = 0
        total_predictions = 0

        for doc_id in document_ids:
            # extract resources
            document = wifine.DOCUMENT_INDEX.get_document(doc_id)

            text = document.sentences_as_tokens()
            text = flatten(text)
            num_tokens = len(text)
            starts, ends = get_word_start_end_positions(text)
            text = ' '.join(text)

            figer_types = get_entity_token_span_to_figer_types(document)

            # text: str
            # starts: token_idx -> char_idx
            # ends: token_idx -> char_idx
            # figer_types: (token_idx, token_idx) -> figer_type

            # train on figer types
            entity_spans = []
            labels = []
            for (start_token_idx, end_token_idx), figer_types in figer_types.items():
                labels.append(figer_types[0])
                entity_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))

            non_entity_spans = []
            for start_token_idx in range(num_tokens):
                for end_token_idx in range(num_tokens):
                    if (start_token_idx, end_token_idx) not in figer_types:
                        non_entity_spans.append((starts[start_token_idx], ends[end_token_idx - 1]))
            non_entity_spans = random.choices(non_entity_spans, k=len(entity_spans))
            labels += [NO_ENTITY_TYPE] * len(non_entity_spans)

            assert len(labels) == len(entity_spans) + len(non_entity_spans)

            for entity_span, label in zip(entity_spans + non_entity_spans, labels):
                inputs = tokenizer(
                    text, 
                    entity_spans=[entity_span],
                    return_tensors='pt'
                )
                outputs = model(**inputs)
                prediction = outputs.logits.argmax(-1).item()

                total_predictions += 1
                if prediction == label:
                    num_correct += 1

        print(num_correct)
        print(total_predictions)

    util.save_checkpoint(model, opt, NUM_EPOCHS)

