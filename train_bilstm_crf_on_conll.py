import logging
import random

import datasets
import torch

import glove_util
import util
from bilstm_crf_util import (
    BiLstmCrfModel,
    make_stats,
    backprop,
    make_inputs, print_eval
)

import allennlp.modules.conditional_random_field as crf


def main():
    util.init_logging()

    CONLL_TRAIN = datasets.load_dataset('conll2003')['train']
    train_dataset = CONLL_TRAIN
    CONLL_VALID = datasets.load_dataset('conll2003')['validation']
    ner_feature = CONLL_TRAIN.features['ner_tags'].feature

    embedding_dim = 50
    embeddings, token_to_idx = glove_util.load_embeddings_tensor_and_token_to_idx_dict(
        dim=embedding_dim
    )
    constraints = crf.allowed_transitions(
        'BIO',
        {i: tag for i, tag in enumerate(ner_feature.names) }
    )
    model = BiLstmCrfModel(
        vocab_size=len(token_to_idx) + 1, # plus one for unk
        num_tags=ner_feature.num_classes,
        embedding_dim=embedding_dim,
        embeddings=embeddings,
        lstm_hidden_dim=50,
        lstm_num_layers=1,
        dropout=0,
        crf_constraints=constraints,
    )
    logging.info(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logging.info(opt)

    num_epochs = 10
    batch_size = 16

    for _ in util.mytqdm(range(num_epochs)):
        train_dataset = train_dataset.shuffle()

        stats = make_stats()

        for batch_idx in util.mytqdm(range(0, len(train_dataset), batch_size)):
            opt.zero_grad()
            examples = train_dataset.select(range(batch_idx, batch_idx + batch_size))
            backprop(
                model,
                [example['tokens'] for example in examples],
                [example['ner_tags'] for example in examples],
                token_to_idx,
                stats
            )
            opt.step()

        logging.info(stats)

        model.eval()
        num_correct = 0
        total = 0

        for example in util.mytqdm(CONLL_VALID):
            inputs = make_inputs(
                [example['tokens']],
                [example['ner_tags']],
                token_to_idx,
                decode_tags=True
            )

            outputs = model(**inputs)
            viterbi_decode = outputs['tags']
            predictions = viterbi_decode[0][0]
            assert len(predictions) == len(example['ner_tags'])
            num_correct += sum(1 for p, a in zip(predictions, example['ner_tags']) if p == a)
            total += len(example['tokens'])

        logging.info(f'acid test num_correct={num_correct}, total={total}')

    # util.save_checkpoint(model, opt, num_epochs)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(e)
        raise e
