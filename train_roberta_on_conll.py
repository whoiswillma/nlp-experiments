import argparse
import logging
import datasets
from transformers import get_scheduler
import torch
import torch.optim as optim

import roberta_util
import conll_util
import util


def do_training(args):

    # tokenizer
    tokenizer = roberta_util.make_tokenizer()

    # dataset
    CONLL_DATASET = roberta_util.encode(
        datasets.load_dataset('conll2003'), tokenizer)
    CONLL_TRAIN = CONLL_DATASET['train']
    train_data = torch.utils.data.DataLoader(CONLL_TRAIN, batch_size=8)

    # model and tokenizer

    num_labels = conll_util.num_labels(CONLL_TRAIN)
    id2label, label2id = conll_util.get_label_mappings(CONLL_TRAIN)
    model = roberta_util.make_model(num_labels, id2label, label2id)

    # optimizer
    DEVICE = util.PTPU
    model.train().to(DEVICE)
    optimizer = optim.AdamW(params=model.parameters(), lr=5e-5)

    # scheduler
    warmup_ratio = 0.1
    total_steps = len(train_data)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    logging.debug(f'starting training with {args.epochs} epochs')
    # iterate through the data 'n_epochs' times
    for epoch in util.mytqdm(range(args.epochs)):
        logging.debug(f'epoch number: {epoch}')

        current_loss = 0
        # iterate through each batch of the train data
        for i, batch in enumerate(util.mytqdm(train_data)):

            # move the batch tensors to the same device as the
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(**batch)
            # the outputs are of shape (loss, logits)
            loss = outputs[0]
            loss.backward()

            current_loss += loss.item()
            # update the model using the optimizer
            optimizer.step()
            lr_scheduler.step()
            # once we update the model we set the gradients to zero
            optimizer.zero_grad()

            logging.debug(f'Train loss at batch {i}: {current_loss}')
            current_loss = 0

        logging.debug(f'saving checkpoint...')
        util.save_checkpoint(model, optimizer, epoch)
        logging.debug(f'done')


def main(args):
    util.init_logging()
    # util.pytorch_set_num_threads(1)

    NUM_EPOCHS = args.epochs

    # # lr from SUMMARY paper

    do_training(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RoBERTa on CoNLL')
    parser.add_argument(
        '--checkpoint', help='path of checkpoint to load', default=None, type=str)
    parser.add_argument('--epochs', help='number of epochs',
                        default=1, type=int)

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logging.warning(e)
        raise e
