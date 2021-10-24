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

from transformers import get_linear_schedule_with_warmup

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
    # prepare the dataset
    conll_datasets = datasets.load_dataset('conll2003')
    conll_train = conll_datasets['train'].map(map_example)
    train_dataloader = DataLoader(
        conll_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )

    # set up model, tokenizer, opt, and scheduler
    model, tokenizer = luke_util.make_model_and_tokenizer(5)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adamw_beta1, args.adamw_beta2),
        eps=args.adamw_eps,
        weight_decay=args.adamw_weight_decay
    )

    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = util.load_checkpoint(
            args.checkpoint,
            into_model=model,
            into_opt=opt
        )
        assert checkpoint['epoch'] >= 0
        start_epoch = checkpoint['epoch'] + 1

    num_train_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = args.scheduler_warmup_ratio * num_train_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
        last_epoch=start_epoch-1
    )

    logging.debug(f'starting epoch: {start_epoch}')
    logging.debug(f'opt = {opt}')
    logging.debug(f'scheduler: {scheduler}')

    # start training
    for epoch in util.mytqdm(range(start_epoch, args.epochs)):
        stats = luke_util.make_train_stats_dict()

        for batch in util.mytqdm(train_dataloader, desc='train'):
            opt.zero_grad()

            logging.debug(f'This batch\'s examples {[example["id"] for example in batch]}')

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
            scheduler.step()

        logging.info(f'stats = {stats}')
        util.save_checkpoint(model, opt, epoch)


def validate(args):
    assert args.checkpoint is not None, 'Must provide checkpoint file when validating'

    model, tokenizer = luke_util.make_model_and_tokenizer(5)
    checkpoint = util.load_checkpoint(
        args.checkpoint,
        into_model=model
    )
    epoch = checkpoint['epoch']

    logging.info(f'Validating model on epoch {epoch}')

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

    logging.info(f'args: {args}')

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

    # LUKE paper Table 12
    parser.add_argument('--learning-rate', help='learning rate', default=1e-5, type=float)
    parser.add_argument('--adamw-beta1', default=0.9, type=float)
    parser.add_argument('--adamw-beta2', default=0.98, type=float)
    parser.add_argument('--adamw-epsilon', default=1e-6, type=float)
    parser.add_argument('--adamw-weight-decay', default=0.01, type=float)
    parser.add_argument('--scheduler-warmup-ratio', default=0.06, type=float)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.warning(e)
        raise e
