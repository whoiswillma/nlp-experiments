import argparse
import logging

import torch
import torch.optim as optim

import roberta_util
import util
import fewnerd_util

from transformers import get_scheduler
from fewnerdparse.dataset import FEWNERD_SUPERVISED, FEWNERD_COARSE_FINE_TYPES

DEVICE = None


def train(args):
    tokenizer = roberta_util.make_tokenizer()

    fewnerd_labels = [fst + "-" + scd for fst, scd in FEWNERD_COARSE_FINE_TYPES]
    num_labels, label2id, id2label = fewnerd_util.labels_to_mappings(fewnerd_labels)

    FEWNERD_TRAIN = roberta_util.encode_fewnerd(
        FEWNERD_SUPERVISED, tokenizer, label2id
    )["train"]

    # FEWNERD_TRAIN = roberta_util.encode_fewnerd(
    #     {
    #         "train": FEWNERD_SUPERVISED["train"][:10],
    #         "dev": FEWNERD_SUPERVISED["dev"][:10],
    #         "test": FEWNERD_SUPERVISED["test"][:10],
    #     },
    #     tokenizer,
    #     label2id,
    # )["train"]

    model = roberta_util.make_model(num_labels, id2label, label2id)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    logging.debug(f"opt = {optimizer}")

    NUM_EPOCHS = args.epochs

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train().to(DEVICE)

    train_data = torch.utils.data.DataLoader(FEWNERD_TRAIN, batch_size=args.batch_size)
    total_steps = len(train_data) / args.grad_acc_steps * NUM_EPOCHS

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    TRAIN_LOSS = []

    for epoch in util.mytqdm(range(NUM_EPOCHS)):
        logging.info(
            f"=====================current epoch: {epoch}====================="
        )
        current_loss = 0

        for i, batch in enumerate(util.mytqdm(train_data)):
            try:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs[0]
                loss.backward()

                current_loss += loss.item()
                logging.info(f"batch_num: {i}, current loss: {current_loss}")

                if i % args.batch_size == 0 and i > 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    TRAIN_LOSS.append(current_loss / 32)
                    current_loss = 0

            except RuntimeError as e:
                util.free_memory()
                logging.warning(e)
                logging.warning(f"Example {i}: {e}")
                continue

        optimizer.step()
        optimizer.zero_grad()

    logging.info(f"Saving checkpoint at epoch: {epoch + 1}")
    util.save_checkpoint(model, optimizer, epoch)


def evaluate(args):
    assert args.checkpoint is not None, "Must provide checkpoint file when validating"

    tokenizer = roberta_util.make_tokenizer()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fewnerd_labels = [fst + "-" + scd for fst, scd in FEWNERD_COARSE_FINE_TYPES]
    num_labels, label2id, id2label = fewnerd_util.labels_to_mappings(fewnerd_labels)

    model = roberta_util.make_model(num_labels, id2label, label2id)

    if args.op == "validate":
        FEWNERD_VAL = roberta_util.encode_fewnerd(
            FEWNERD_SUPERVISED, tokenizer, label2id
        )["dev"]

    else:
        FEWNERD_VAL = roberta_util.encode_fewnerd(
            FEWNERD_SUPERVISED, tokenizer, label2id
        )["test"]

    val_data = torch.utils.data.DataLoader(FEWNERD_VAL, batch_size=args.batch_size)

    checkpoint = util.load_checkpoint(args.checkpoint, model=model)
    epoch = checkpoint["epoch"]
    logging.info(f"Validating model on epoch {epoch}")

    confusion_matrix = torch.zeros(num_labels, num_labels)
    model.eval().to(DEVICE)

    for i, batch in enumerate(util.mytqdm(val_data)):
        with torch.no_grad():
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)

        s_lengths = batch["attention_mask"].sum(dim=1)

        # logging.info(f"outputs: {outputs}, s_lengths: {s_lengths}")

        for idx, length in enumerate(s_lengths):
            true_val = batch["labels"][idx][:length]
            pred_val = torch.argmax(outputs[1], dim=2)[idx][:length]

            # if i < 5:
            #     logging.info(f"true_val:{true_val}, pred_val:{pred_val}")

            for true, pred in zip(true_val, pred_val):
                confusion_matrix[true.item()][pred.item()] += 1

    precisions = []
    recalls = []
    f_1s = []

    torch.set_printoptions(profile="full")

    sum_cols, sum_rows = confusion_matrix.sum(0), confusion_matrix.sum(1)
    diagonals = torch.diagonal(confusion_matrix, 0)

    logging.info(f"sum_cols: {sum_cols}, sum_rows: {sum_rows}")

    for i, tp in enumerate(diagonals):
        precision = tp / sum_cols[i]
        recall = tp / sum_rows[i]
        f_1 = 2 * (precision * recall) / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f_1s.append(f_1)

        logging.info(f"precision: {precision}, recall: {recall}, f1: {f_1}")

    avg_precision = sum(precisions) / num_labels
    avg_recall = sum(recalls) / num_labels
    avg_f1 = sum(f_1s) / num_labels
    accuracy = sum(diagonals) / torch.sum(confusion_matrix)

    logging.info(f"avg precision: {avg_precision}")
    logging.info(f"avg recall: {avg_recall}")
    logging.info(f"avg f1: {avg_f1}")
    logging.info(f"accuracy: {accuracy}")

    # logging.info(f"confusion_matrix: {confusion_matrix}")


def main(args):
    util.init_logging()
    logging.info(f"args: {args}")

    if args.op == "train":
        train(args)
    elif args.op == "validate" or args.op == "test":
        evaluate(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa on fewNERD")
    parser.add_argument(
        "op",
        help="operation to perform",
        default="train",
        choices=["train", "validate", "test"],
    )

    parser.add_argument(
        "--checkpoint", help="path of checkpoint to load", default=None, type=str
    )
    parser.add_argument("--batch_size", help="train batch size", default=4, type=int)
    parser.add_argument(
        "--grad_acc_steps",
        help="Number of training steps for which the gradients should be accumulated.",
        default=2,
        type=int,
    )

    parser.add_argument("--epochs", help="number of epochs", default=5, type=int)
    parser.add_argument(
        "--learning_rate", help="learning rate", default=1e-5, type=float
    )
    parser.add_argument("--warmup_ratio", help="warmup ratio", default=0.1, type=int)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.warning(e)
        raise e
