import logging
import datasets
from transformers import get_scheduler
import torch
import torch.optim as optim

import roberta_util
import luke_util
import util
from fewnerdparse.dataset import FEWNERD_SUPERVISED, FEWNERD_COARSE_FINE_TYPES

def do_training(model, n_epochs, train_data, optimizer, effective_batch_size=16, warmup_ratio=0.1):

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

def get_entity_spans_to_label(example) -> dict[tuple[int, int]: int]:
    entity_spans_to_labels: dict[tuple[int, int]: int] = {}

    outside_label = len(FEWNERD_COARSE_FINE_TYPES)

    tokens = example['tokens']
    label_ids: list[int] = [
        FEWNERD_COARSE_FINE_TYPES.index((coarse, fine))
        if coarse and fine else outside_label
        for coarse, fine in zip(example['coarse_labels'], example['fine_labels'])
    ]

    current_entity_start = -1
    current_label = outside_label

    for i, (token, label) in enumerate(list(zip(tokens, label_ids)) + [('', outside_label)]):
        if current_label != label:
            if current_label != outside_label:
                assert 0 <= current_entity_start
                entity_spans_to_labels[(current_entity_start, i)] = current_label

            current_entity_start = i

        current_label = label

    return entity_spans_to_labels


def main():
    util.init_logging()
    # util.pytorch_set_num_threads(1)

    tokenizer = roberta_util.make_tokenizer()
    model = roberta_util.make_model(num_labels, id2label, label2id)

    NONENTITY_LABEL = len(FEWNERD_COARSE_FINE_TYPES)
    NUM_EPOCHS = 5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train().to(DEVICE)

    optimizer = optim.AdamW(params=model.parameters(), lr=5e-5)
    logging.debug(f'opt = {optimizer}')

    FEWNERD_TRAIN = FEWNERD_SUPERVISED['train'][:10]

    # for epoch in util.mytqdm(range(NUM_EPOCHS)):
    #     stats = luke_util.make_train_stats_dict()

    #     for example in util.mytqdm(FEWNERD_TRAIN, desc='train'):
    #         opt.zero_grad()

    #         entity_spans_to_labels = get_entity_spans_to_label(example)

    #         luke_util.train_luke_model(
    #             model,
    #             tokenizer,
    #             example['tokens'],
    #             entity_spans_to_labels,
    #             nonentity_label=NONENTITY_LABEL,
    #             stats=stats
    #         )

    #         opt.step()

    #     logging.info(f'stats = {stats}')
    #     # util.save_checkpoint(model, opt, epoch)

    #     # validate
    #     correct = 0
    #     total = 0

    #     for example in util.mytqdm(FEWNERD_TRAIN, desc='validate'):
    #         entity_spans_to_labels = get_entity_spans_to_label(example)

    #         doc_correct, doc_total = luke_util.acid_test_luke_model(
    #             model,
    #             tokenizer,
    #             example['tokens'],
    #             entity_spans_to_labels=entity_spans_to_labels,
    #             nonentity_label=NONENTITY_LABEL
    #         )

    #         correct += doc_correct
    #         total += doc_total

    #     logging.info('Validation')
    #     logging.info(f'num_correct = {correct}')
    #     logging.info(f'total_predictions = {total}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.warning(e)
        raise e
