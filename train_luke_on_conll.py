import argparse
import logging
from typing import Optional

import datasets
import torch
import torch.utils.data
from torch.utils.data import DataLoader

import conll_util
import luke_util
import ner
import util

CONLL_TO_LABEL_MAP = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 4,
    8: 4
}

LABEL_TO_STR_MAP = {
    0: 'O',
    1: 'PER',
    2: 'ORG',
    3: 'LOC',
    4: 'MISC'
}

nonentity_label = 0


def map_example(example):
    example['labels'] = [CONLL_TO_LABEL_MAP[x] for x in example['ner_tags']]
    return example


def get_entity_spans_to_label(labels: list[int]):
    entity_spans_to_label = {}

    prev = 0
    entity_label = None
    entity_start = None

    for idx, label in enumerate(labels + [0]):
        if label != prev:
            if prev != 0:
                entity_spans_to_label[(entity_start, idx)] = entity_label

            entity_start = idx
            entity_label = label

        prev = label

    return entity_spans_to_label


def train(args):
    model, tokenizer = luke_util.make_model_and_tokenizer(5)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.checkpoint is not None:
        util.load_checkpoint(args.checkpoint, model, opt)

    logging.debug(f'opt = {opt}')

    conll_datasets = datasets.load_dataset('conll2003')
    conll_train = conll_datasets['train'].map(map_example)
    train_dataloader = DataLoader(
        conll_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )

    for epoch in util.mytqdm(range(args.epochs)):
        stats = luke_util.make_train_stats_dict()

        for batch in util.mytqdm(train_dataloader, desc='train'):
            opt.zero_grad()

            for example in batch:
                try:
                    entity_spans_to_labels = get_entity_spans_to_label(example['labels'])

                    luke_util.train_luke_model(
                        model,
                        tokenizer,
                        example['tokens'],
                        entity_spans_to_labels,
                        nonentity_label,
                        stats
                    )

                except RuntimeError as e:
                    logging.warning(e)
                    logging.warning(f'index: {example["id"]}, example: {example}')
                    logging.warning('Moving onto the next training example for now...')
                    logging.warning('')

                    util.free_memory()

            opt.step()

        logging.info(f'stats = {stats}')
        util.save_checkpoint(model, opt, epoch)


def validate(args):
    assert args.checkpoint is not None, 'Must provide checkpoint file when validating'

    model, tokenizer = luke_util.make_model_and_tokenizer(5)
    util.load_checkpoint(
        args.checkpoint,
        model=model
    )

    conll_datasets = datasets.load_dataset('conll2003')
    conll_valid = conll_datasets['validation'].map(map_example)
    label2id, id2label = conll_util.get_label_mappings(conll_valid)

    confusion_matrix = ner.NERBinaryConfusionMatrix()
    for example in util.mytqdm(conll_valid, desc='validate'):
        predictions = luke_util.eval_named_entity_spans(
            model,
            tokenizer,
            example['tokens'],
            nonentity_label,
            16
        )
        predictions = { LABEL_TO_STR_MAP[idx]: spans for idx, spans in predictions.items() }

        gold = list(map(id2label.get, example['ner_tags']))
        gold = ner.extract_named_entity_spans_from_bio(gold)

        ner.compute_binary_confusion_from_named_entity_spans(predictions, gold, confusion_matrix)

    logging.info('Validation')
    logging.info(f'Confusion {confusion_matrix}')


def main(args):
    util.init_logging()

    logging.info(args)

    if args.op == 'train':
        train(args)
    elif args.op == 'validate':
        validate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LUKE on CoNLL')
    parser.add_argument('op', help='operation to perform', default='train', choices=['train', 'validate'])
    parser.add_argument('--checkpoint', help='path of checkpoint to load', default=None, type=Optional[str])
    parser.add_argument('--batch-size', help='train batch size', default=8, type=int)
    parser.add_argument('--epochs', help='number of epochs', default=5, type=int)
    parser.add_argument('--learning-rate', help='learning rate', default=1e-5, type=float)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.warning(e)
        raise e
