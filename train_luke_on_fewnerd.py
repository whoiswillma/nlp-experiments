import argparse
import logging

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import luke_util
import ner
import util
import fewnerdparse.dataset as dataset
from fewnerdparse.dataset import FEWNERD_COARSE_FINE_TYPES, load_dataset


# the idx of the 'O' label
NONENTITY_LABEL = FEWNERD_COARSE_FINE_TYPES.index(('O', 'O'))


def get_entity_spans_to_label(example) -> dict[tuple[int, int]: int]:
    entity_spans_to_labels: dict[tuple[int, int]: int] = {}

    tokens = example['tokens']
    label_ids: list[int] = [
        FEWNERD_COARSE_FINE_TYPES.index((coarse, fine))
        for coarse, fine in zip(example['coarse_labels'], example['fine_labels'])
    ]

    current_entity_start = -1
    current_label = NONENTITY_LABEL

    for i, (token, label) in enumerate(list(zip(tokens, label_ids)) + [('', NONENTITY_LABEL)]):
        if current_label != label:
            if current_label != NONENTITY_LABEL:
                assert 0 <= current_entity_start
                entity_spans_to_labels[(current_entity_start, i)] = current_label

            current_entity_start = i

        current_label = label

    return entity_spans_to_labels


def train(args):
    fewnerd_train = load_dataset(args.dataset_split, 'train')

    if args.dataset_scale != 1:
        s = args.dataset_scale
        logging.info('Scaling the train dataset by {args.dataset_scale}')
        count = int(len(fewnerd_train) * s)
        logging.info('Taking the first {count} examples of the training dataset')
        fewnerd_train = fewnerd_train[:count]

    train_dataloader = DataLoader(
        fewnerd_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x
    )

    # set up model, tokenizer, opt, and scheduler
    model, tokenizer = luke_util.make_model_and_tokenizer(len(FEWNERD_COARSE_FINE_TYPES))
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
            model=model,
            opt=opt
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

            for example in batch:
                try:
                    entity_spans_to_labels = get_entity_spans_to_label(example)

                    luke_util.train_luke_model(
                        model,
                        tokenizer,
                        example['tokens'],
                        entity_spans_to_labels,
                        nonentity_label=NONENTITY_LABEL,
                        stats=stats,
                        example_id=example['id']
                    )

                except RuntimeError as e:
                    util.free_memory()

                    logging.warning('')
                    logging.warning(e)
                    logging.warning(f'Example: {example}')
                    logging.warning('Moving onto the next training example for now...')
                    logging.warning('')


            opt.step()
            scheduler.step()

        logging.info(f'stats = {stats}')
        util.save_checkpoint(model, opt, epoch)


def evaluate(args):
    assert args.checkpoint is not None, 'Must provide checkpoint file when validating'

    model, tokenizer = luke_util.make_model_and_tokenizer(len(FEWNERD_COARSE_FINE_TYPES) + 1)
    checkpoint = util.load_checkpoint(
        args.checkpoint,
        model=model
    )
    epoch = checkpoint['epoch']

    logging.info(f'Validating/testing model on epoch {epoch}')

    if args.op == 'validate':
        eval_dataset = load_dataset(args.dataset_split, 'dev')
    else:
        assert args.op == 'test'
        eval_dataset = load_dataset(args.dataset_split, 'test')

    confusion_matrix = ner.NERBinaryConfusionMatrix()
    for example in util.mytqdm(eval_dataset, desc='validate'):
        predictions = luke_util.eval_named_entity_spans(
            model,
            tokenizer,
            example['tokens'],
            NONENTITY_LABEL,
            16
        )
        predictions = {
            dataset.recombine(*FEWNERD_COARSE_FINE_TYPES[idx]): spans
            for idx, spans in predictions.items()
        }

        gold = [
            dataset.recombine(coarse, fine)
            for coarse, fine in zip(example['coarse_labels'], example['fine_labels']) 
            if (coarse, fine) != ('O', 'O')
        ]
        gold = ner.extract_named_entity_spans_from_chunks(gold)

        ner.compute_binary_confusion_from_named_entity_spans(predictions, gold, confusion_matrix)

    if args.op == 'validate':
        logging.info(f'On FEWNERD {args.dataset_split} DEV:')
    else:
        logging.info(f'On FEWNERD {args.dataset_split} TEST:')
    logging.info(f'Confusion {confusion_matrix}')


def main(args):
    util.init_logging()

    logging.info('Depending on the operation being performed, not all args may be relevant.')
    logging.info(f'args: {args}')

    if args.op == 'train':
        train(args)
    else:
        assert args.op in {'validate', 'test'}
        evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LUKE on FewNERD')
    parser.add_argument('op', help='operation to perform', default='train', choices=['train', 'validate', 'test'])
    parser.add_argument('--checkpoint', help='path of checkpoint to load', default=None, type=str)
    parser.add_argument('--batch-size', help='train batch size', default=8, type=int)
    parser.add_argument('--epochs', help='number of epochs', default=5, type=int)

    # LUKE paper Table 12
    parser.add_argument('--learning-rate', help='learning rate', default=1e-5, type=float)
    parser.add_argument('--adamw-beta1', default=0.9, type=float)
    parser.add_argument('--adamw-beta2', default=0.98, type=float)
    parser.add_argument('--adamw-eps', default=1e-6, type=float)
    parser.add_argument('--adamw-weight-decay', default=0.01, type=float)
    parser.add_argument('--scheduler-warmup-ratio', default=0.06, type=float)

    # the amount to "scale" the dataset by, e.g. 0.01 would train/eval on only
    # 1% of the dataset.
    parser.add_argument('--dataset-scale', default=1, type=float)
    parser.add_argument('--dataset-split', default='supervised', choices=['supervised', 'intra', 'inter'])

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.warning(e)
        raise e
