import logging

import datasets
import torch

import glove_util
import util
from bilstm_crf_util import (
    BiLstmCrfModel,
    make_stats,
    backprop,
    make_inputs
)

import allennlp.modules.conditional_random_field as crf


def main():
    util.init_logging()

    CONLL_TRAIN = datasets.load_dataset('conll2003')['train']
    CONLL_VALID = datasets.load_dataset('conll2003')['validation']
    ner_feature = CONLL_TRAIN.features['ner_tags'].feature

    constraints = crf.allowed_transitions(
        'BIO',
        {i: tag for i, tag in enumerate(ner_feature.names) }
    )
    embeddings, token_to_idx = glove_util.load_embeddings_tensor_and_token_to_idx_dict()
    model = BiLstmCrfModel(
        vocab_size=len(token_to_idx) + 1, # plus one for unk
        num_tags=ner_feature.num_classes,
        embeddings=embeddings,
        crf_constraints=constraints
    )
    logging.info(model)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logging.info(opt)

    num_epochs = 20
    batch_size = 16

    for _ in util.mytqdm(range(num_epochs), desc='epoch'):
        stats = make_stats()

        for batch_idx in util.mytqdm(range(0, len(CONLL_TRAIN), batch_size), desc='batch'):
            opt.zero_grad()
            examples = CONLL_TRAIN.select(range(batch_idx, batch_idx + batch_size))
            backprop(model, examples, token_to_idx, stats)
            opt.step()

        logging.info(stats)

        model.eval()
        num_correct = 0
        total = 0

        for example in util.mytqdm(CONLL_VALID, desc='valid'):
            inputs = make_inputs([example], token_to_idx, decode_tags=True)
            viterbi_decode = model(**inputs)['tags']
            predictions = viterbi_decode[0][0]
            num_correct += sum(1 for p, a in zip(predictions, example['ner_tags']) if p == a)
            total += len(example['tokens'])

        logging.info(f'acid test num_correct={num_correct}, total={total}')

    util.save_checkpoint(model, opt, num_epochs)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.warning(e)
        raise e
