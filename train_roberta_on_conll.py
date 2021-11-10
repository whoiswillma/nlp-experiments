import logging
import datasets
from transformers import get_scheduler
import torch
import torch.optim as optim

import roberta_util
import conll_util
import util

TRAIN_LOSS = []
DEVICE = None
SAVE_PATH = "checkpoints"


def do_training(
    model, n_epochs, train_data, optimizer, effective_batch_size=16, warmup_ratio=0.1
):

    grad_acc_steps = effective_batch_size // 4
    total_steps = len(train_data) / grad_acc_steps * n_epochs

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    num_skips = 0

    print("Beginning Training:")
    # iterate through the data 'n_epochs' times
    for epoch in util.mytqdm(range(n_epochs)):
        print("Epoch: ", epoch + 1)
        current_loss = 0
        # iterate through each batch of the train data
        for i, batch in enumerate(util.mytqdm(train_data)):

            if epoch < num_skips:
                lr_scheduler.step()
            else:
                # move the batch tensors to the same device as the
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                # send 'input_ids', 'attention_mask' and 'labels' to the model
                outputs = model(**batch)
                # the outputs are of shape (loss, logits)
                loss = outputs[0]
                loss.backward()

                current_loss += loss.item()
                if i % 4 == 0 and i > 0:
                    # update the model using the optimizer
                    optimizer.step()
                    lr_scheduler.step()
                    # once we update the model we set the gradients to zero
                    optimizer.zero_grad()
                    # store the loss value for visualization
                    TRAIN_LOSS.append(current_loss / 32)
                    current_loss = 0

        # SAVING CHECKPOINTS!!!
        print("saving...")
        if epoch >= num_skips:
            # update the model one last time for this epoch
            optimizer.step()
            optimizer.zero_grad()
            path = SAVE_PATH + "/" + str(epoch + 1) + "e"
            print(path)
            model.save_pretrained(path)
            print("Saving checkpoint at epoch: ", epoch + 1)


def main():
    util.init_logging()
    # util.pytorch_set_num_threads(1)

    tokenizer = roberta_util.make_tokenizer()

    CONLL_DATASET = roberta_util.encode(datasets.load_dataset("conll2003"), tokenizer)
    CONLL_TRAIN = CONLL_DATASET["train"]
    CONLL_TRAIN_10 = conll_util.downsample(CONLL_TRAIN, 10)
    CONLL_TRAIN_1 = conll_util.downsample(CONLL_TRAIN, 1)

    num_labels = conll_util.num_labels(CONLL_TRAIN)
    id2label, label2id = conll_util.get_label_mappings(CONLL_TRAIN)
    model = roberta_util.make_model(num_labels, id2label, label2id)

    NUM_EPOCHS = 3

    # # lr from SUMMARY paper
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train().to(DEVICE)
    optimizer = optim.AdamW(params=model.parameters(), lr=5e-5)
    logging.debug(f"opt = {optimizer}")

    train_data = torch.utils.data.DataLoader(CONLL_TRAIN_1, batch_size=4)
    do_training(model, 1, train_data, optimizer)

    #     logging.info('Validation')
    #     logging.info(f'num_correct = {correct}')
    #     logging.info(f'total_predictions = {total}')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.warning(e)
        raise e
