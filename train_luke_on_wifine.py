import logging
import time

import torch

import util
import luke_util
from wifineparse import wifine


def flatten(ll: list[list]):
    return [e for l in ll for e in l]


def get_document_ids_to_train_with(num=None, fraction=None):
    if fraction:
        num = int(fraction * len(wifine.DOCUMENT_INDEX.all_docids()))

    return wifine.DOCUMENT_INDEX.all_docids()[:num]


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
    if fe:
        return {
            (sent_offs[s] + i, sent_offs[s] + j): t
            for i, j, t, s in zip(fe.begin, fe.end, fe.figer_types, fe.sent_idx)
        }
    else:
        return {}


def get_entity_token_idx_to_figer_type(doc):
    sent_offs = get_sentence_offsets(doc)
    fe = doc.fine_entities()
    if fe:
        return {
            sent_offs[s] + i: t[0]
            for start, end, t, s in zip(fe.begin, fe.end, fe.figer_types, fe.sent_idx)
            for i in range(start, end)
        }
    else:
        return {}


def main():
    util.init_logging()
    util.pytorch_set_num_threads(1)

    # make luke model and tokenizer
    model, tokenizer = luke_util.make_model_and_tokenizer(
        num_labels=len(wifine.FIGER_VOCAB) + 1
    )

    # the label that represents no entity type (i.e. 'O')
    NONENTITY_LABEL = len(wifine.FIGER_VOCAB)

    # get train dataset
    train_document_ids = get_document_ids_to_train_with(num=100)
    valid_document_ids = train_document_ids  # for testing purposes
    logging.debug(f"train_document_ids = {train_document_ids}")
    logging.debug(f"valid_document_ids = {valid_document_ids}")

    # learning setup
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    logging.debug(f"opt = {opt}")

    NUM_EPOCHS = 10

    for epoch in util.mytqdm(range(1, NUM_EPOCHS + 1), desc="epochs"):
        stats = luke_util.make_train_stats_dict()

        for doc_id in util.mytqdm(train_document_ids, desc="train"):
            opt.zero_grad()

            # extract resources
            document = wifine.DOCUMENT_INDEX.get_document(doc_id)
            tokens = flatten(document.sentences_as_tokens())

            figer_types = get_entity_token_span_to_figer_types(document)
            # choose just one figer type for each entity span
            # TODO: how should we incorporate multiple figer types into pretraining?
            figer_types = {k: v[0] for k, v in figer_types.items()}

            did_backprop = False
            try:
                luke_util.train_luke_model(
                    model,
                    tokenizer,
                    tokens,
                    entity_spans_to_labels=figer_types,
                    nonentity_label=NONENTITY_LABEL,
                    stats=stats,
                )
                did_backprop = True

            except ValueError:
                logging.debug(
                    f"Document {doc_id} is too long. Trying again with"
                    + " nonentity_choose_k='num_entity_spans'"
                )

            if not did_backprop:
                try:
                    luke_util.train_luke_model(
                        model,
                        tokenizer,
                        tokens,
                        entity_spans_to_labels=figer_types,
                        nonentity_label=NONENTITY_LABEL,
                        stats=stats,
                        nonentity_choose_k="num_entity_spans",
                    )
                    did_backprop = True

                except ValueError:
                    logging.debug(f"Document {doc_id} is too long. Skipping")

            if did_backprop:
                opt.step()

        logging.info(f"stats = {stats}")
        # util.save_checkpoint(model, opt, epoch)

        # validate
        correct = 0
        total = 0

        for doc_id in util.mytqdm(valid_document_ids, desc="validate"):
            document = wifine.DOCUMENT_INDEX.get_document(doc_id)
            tokens = flatten(document.sentences_as_tokens())

            figer_types = get_entity_token_span_to_figer_types(document)
            figer_types = {k: v[0] for k, v in figer_types.items()}

            doc_correct, doc_total = luke_util.acid_test_luke_model(
                model,
                tokenizer,
                tokens,
                entity_spans_to_labels=figer_types,
                nonentity_label=NONENTITY_LABEL,
            )

            correct += doc_correct
            total += doc_total

        logging.info("Validation")
        logging.info(f"num_correct = {correct}")
        logging.info(f"total_predictions = {total}")

    # checkpoint_name = util.save_checkpoint(model, opt, NUM_EPOCHS)
    # logging.info(f'Saved checkpoint {checkpoint_name}')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.warning(e)
        raise e
