import logging

import datasets
import torch
from tqdm import tqdm

import luke_util
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


def map_to_int_labels(example):
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


if __name__ == '__main__':
    util.init_logging()
    util.pytorch_set_num_threads(1)

    model, tokenizer = luke_util.make_model_and_tokenizer(5)

    CONLL_DATASET = datasets.load_dataset('conll2003')
    CONLL_TRAIN = CONLL_DATASET['train'].map(map_to_int_labels)
    # CONLL_VALID = CONLL_DATASET['validation'].map(map_to_int_labels)
    # CONLL_TEST = CONLL_DATASET['test'].map(map_to_int_labels)

    NUM_EPOCHS = 1

    # lr from LUKE paper
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    logging.debug(f'opt = {opt}')

    for epoch in util.mytqdm(range(NUM_EPOCHS)):
        stats = luke_util.make_train_stats_dict()

        for example in util.mytqdm(CONLL_TRAIN):
            opt.zero_grad()
            entity_spans_to_labels = get_entity_spans_to_label(example['labels'])

            luke_util.train_luke_model(
                model,
                tokenizer,
                example['tokens'],
                entity_spans_to_labels,
                0,
                stats
            )
            opt.step()

        logging.info(f'stats = {stats}')
        util.save_checkpoint(model, opt, epoch)

