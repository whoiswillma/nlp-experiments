import logging

import datasets
import allennlp.modules.conditional_random_field as crf

import util
from bilstm_crf_util import *


def main():
    util.init_logging()

    CONLL_TRAIN = datasets.load_dataset('conll2003')['train']
    CONLL_VALID = datasets.load_dataset('conll2003')['validation']
    token_to_idx: dict[str, int] = generate_token_to_idx_dict(CONLL_TRAIN)
    ner_feature = CONLL_TRAIN.features['ner_tags'].feature

    constraints = crf.allowed_transitions(
        'BIO',
        {i: tag for i, tag in enumerate(ner_feature.names) }
    )
    model = BiLstmCrfModel(
        vocab_size=len(token_to_idx) + 1, # plus one for unk
        num_tags=ner_feature.num_classes,
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
