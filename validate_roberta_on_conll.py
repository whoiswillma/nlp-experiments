import argparse
import logging
import datasets
from transformers import get_scheduler
import torch
import torch.optim as optim

import roberta_util
import conll_util
import util

import eval


def do_validation(args):

    # tokenizer
    tokenizer = roberta_util.make_tokenizer()

    # dataset
    CONLL_DATASET = roberta_util.encode(
        datasets.load_dataset('conll2003'), tokenizer)
    CONLL_TEST = CONLL_DATASET['test']
    test_data = torch.utils.data.DataLoader(CONLL_TEST, batch_size=8)

    # model and tokenizer

    num_labels = conll_util.num_labels(CONLL_TEST)
    id2label, label2id = conll_util.get_label_mappings(CONLL_TEST)
    model = roberta_util.make_model(num_labels, id2label, label2id)

    # optimizer
    DEVICE = util.PTPU
    model.eval().to(DEVICE)
    # optimizer = optim.AdamW(params=model.parameters(), lr=5e-5)

    # # scheduler
    # warmup_ratio = 0.1
    # total_steps = len(train_data)*args.epochs
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=int(total_steps * warmup_ratio),
    #     num_training_steps=total_steps,
    # )

    # load checkpoint
    logging.debug('loading checkpoint...')
    checkpoint = util.load_checkpoint(
        args.checkpoint,
        model=model
    )
    epoch = checkpoint['epoch']
    logging.debug('checkpoint loaded successfully')

    truth = []
    predictions = []
    TP, FP, FN = 0, 0, 0
    confusion = torch.zeros(num_labels, num_labels)

    # iterate through each batch of the test data
    for i, batch in enumerate(util.mytqdm(test_data)):
        logging.debug(f'running batch {i}')

        # do not calculate the gradients
        with torch.no_grad():
            # move the batch tensors to the same device as the model
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(**batch)

        s_lengths = batch['attention_mask'].sum(dim=1)

        for idx, length in enumerate(s_lengths):
            # get the true values
            true_values = batch['labels'][idx][:length]
            # get the predicted values
            pred_values = torch.argmax(outputs[1], dim=2)[idx][:length]
            # go through all true and predicted values and store them in the confusion matrix
            true_labels, pred_labels = eval.to_label_list(
                true_values, pred_values, id2label)
            truth.append(true_labels)
            predictions.append(pred_labels)

            TP_, FP_, FN_ = eval.calc_scores(true_values, pred_values, id2label)
            TP += TP_
            FP += FP_
            FN += FN_
            # print(TP_, FP_, FN_)
            for true, pred in zip(true_values, pred_values):
                confusion[true.item()][pred.item()] += 1
    f1, precision, recall = eval.get_scores(TP, FP, FN)
    logging.debug(f'TP: {TP}')
    logging.debug(f'FP: {FP}')
    logging.debug(f'FN: {FN}')
    logging.debug(f'Precision: {precision}')
    logging.debug(f'Recall: {recall}')
    logging.debug(f'F1: {f1}')


def main(args):
    util.init_logging()
    # util.pytorch_set_num_threads(1)

    # # lr from SUMMARY paper

    do_validation(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate RoBERTa on CoNLL')
    parser.add_argument(
        '--checkpoint', help='path of checkpoint to load', default=None, type=str)

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logging.warning(e)
        raise e
