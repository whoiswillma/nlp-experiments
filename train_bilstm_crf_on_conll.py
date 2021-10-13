import logging

import allennlp.modules.conditional_random_field as crf
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


def main():
    util.init_logging()

    CONLL_TRAIN = datasets.load_dataset('conll2003')['train']
    train_dataset = CONLL_TRAIN
    CONLL_VALID = datasets.load_dataset('conll2003')['validation']
    ner_feature = CONLL_TRAIN.features['ner_tags'].feature

    embedding_dim = 100
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
        freeze_embeddings=True,
        lstm_hidden_dim=300,
        lstm_num_layers=1,
        dropout=0,
        crf_constraints=constraints,
    )
    logging.info(f'Constraints {constraints}')
    logging.info(model)

    opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    logging.info(opt)

    num_epochs = 50
    batch_size = 100

    for epoch in util.mytqdm(range(1, num_epochs + 1)):
        train_dataset = train_dataset.shuffle()

        stats = make_stats()

        for batch_idx in util.mytqdm(range(0, len(train_dataset) // batch_size)):
            opt.zero_grad()
            examples = train_dataset.select(range(
                batch_size * batch_idx,
                batch_size * (batch_idx + 1)
            ))
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
        validation_loss = 0

        for example in util.mytqdm(CONLL_VALID):
            inputs = make_inputs(
                [example['tokens']],
                [example['ner_tags']],
                token_to_idx,
                decode_tags=True,
                compute_loss=True
            )

            outputs = model(**inputs)
            viterbi_decode = outputs['tags']
            validation_loss += outputs['loss'].item()
            predictions = viterbi_decode[0][0]
            assert len(predictions) == len(example['ner_tags'])
            num_correct += sum(1 for p, a in zip(predictions, example['ner_tags']) if p == a)
            total += len(example['tokens'])

        logging.info(f'validation loss: {validation_loss}')
        logging.info(f'acid test num_correct={num_correct}, total={total}')

        if epoch % 10 == 0:
            util.save_checkpoint(model, opt, epoch)

    util.save_checkpoint(model, opt, epoch)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(e)
        raise e
