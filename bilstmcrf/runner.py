import argparse
import logging
import os
import time
from typing import Dict, List

import allennlp.modules.conditional_random_field as crf
import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PAD, UNK, Conll2003, device
from model import BiLSTM_CRF
from util import (
    build_token_mappings,
    build_tag_mappings,
    calculate_epoch_time,
    compute_entity_level_f1,
    count_parameters,
    pad_batch,
    pad_test_batch,
)


def load_data():
    conll_dataset = datasets.load_dataset("conll2003")
    train = conll_dataset["train"]
    val = conll_dataset["validation"]
    test = conll_dataset["test"]
    return train, val, test


def train_model(model, dataloader, optimizer, clip: int) -> float:
    model.train()
    epoch_loss = 0
    with tqdm(dataloader, unit="batch") as tqdm_loader:
        for x_padded, x_lens, y_padded in tqdm_loader:
            optimizer.zero_grad()
            result = model(x_padded, x_lens, y_padded, decode=False)
            neg_log_likelihood = result["loss"]
            neg_log_likelihood.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += neg_log_likelihood.item()
    return epoch_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader) -> float:
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tqdm_loader:
            for x_padded, x_lens, y_padded in tqdm_loader:
                result = model(x_padded, x_lens, y_padded, decode=False)
                neg_log_likelihood = result["loss"]
                epoch_loss += neg_log_likelihood.item()
    return epoch_loss / len(dataloader.dataset)


def test_model(
    test_data,
    model,
    batch_size: int,
    tokens_to_idx: Dict[str, int],
    idx_to_tags: Dict[int, str],
):
    with torch.no_grad():
        predictions = []
        for batch_idx in range((len(test_data) // batch_size) + 1):
            if (batch_size * (batch_idx + 1)) > len(test_data):
                batch = test_data.select(
                    range(batch_size * (batch_idx), len(test_data))
                )
            else:
                batch = test_data.select(
                    range(batch_size * batch_idx, batch_size * (batch_idx + 1))
                )
            encoded_tokens = []
            for token_seq in batch["tokens"]:
                encoded_seq = []
                for token in token_seq:
                    if token in tokens_to_idx:
                        encoded_seq.append(tokens_to_idx[token])
                    else:
                        encoded_seq.append(tokens_to_idx[UNK])
                encoded_tokens.append(torch.LongTensor(encoded_seq))
            batch_predictions = decode_batch(
                model, encoded_tokens, idx_to_tags=idx_to_tags
            )
            for pred in batch_predictions:
                predictions.append(pred)
    return predictions


def decode_batch(model, batch: List[torch.LongTensor], idx_to_tags: Dict[int, str]):
    model.eval()
    with torch.no_grad():
        padded_batch = pad_test_batch(batch)
        x_padded, x_lens = padded_batch
        result = model(x_padded, x_lens, None, decode=True)
        actual_pred_tags = []
        for pred, _ in result["tags"]:
            actual_pred_tags.append([idx_to_tags[i] for i in pred])
    return actual_pred_tags


def main(args):
    # init logger
    log_ouput = os.path.join(args.out, "logs", f"run_log_{args.run_id}")
    logging.basicConfig(
        filename=log_ouput,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    LOG = logging.getLogger("BiLSTM-CRF")
    LOG.info("BiLSTM-CRF Model on Conll-2003 NER")

    # mappings + datasets + dataloaders
    train, val, test = load_data()
    # test = test.select(range(100))
    ner_tags = train.features["ner_tags"].feature.names
    tokens_to_idx, idx_to_tokens = build_token_mappings(train["tokens"])
    tags_to_idx, idx_to_tags = build_tag_mappings(ner_tags)
    train_data = Conll2003(
        tokens=train["tokens"],
        labels=train["ner_tags"],
        idx_to_tokens=idx_to_tokens,
        tokens_to_idx=tokens_to_idx,
        tags_to_idx=tags_to_idx,
        idx_to_tags=idx_to_tags,
    )
    val_data = Conll2003(
        tokens=val["tokens"],
        labels=val["ner_tags"],
        idx_to_tokens=idx_to_tokens,
        tokens_to_idx=tokens_to_idx,
        tags_to_idx=tags_to_idx,
        idx_to_tags=idx_to_tags,
    )
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_batch,
    )
    val_dataloader = DataLoader(
        dataset=val_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch
    )

    # define model
    crf_constraints = crf.allowed_transitions(
        constraint_type="BIO", labels=train_data.idx_to_tags
    )
    bilstm_crf = BiLSTM_CRF(
        vocab_size=len(train_data.idx_to_tokens.keys()),
        num_tags=len(train_data.idx_to_tags.keys()),
        embedding_dim=args.embedding_dim,
        lstm_hidden_dim=args.hidden_dim,
        lstm_num_layers=args.num_layers,
        dropout=args.dropout,
        constraints=crf_constraints,
        pad_idx=train_data.tokens_to_idx[PAD],
    )
    bilstm_crf.to(device)

    # log number of model params
    num_params = count_parameters(bilstm_crf)
    LOG.info(f"The model has {num_params:,} trainable parameters")

    # run model
    optimizer = torch.optim.Adam(bilstm_crf.parameters())
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_model(
            model=bilstm_crf,
            dataloader=train_dataloader,
            optimizer=optimizer,
            clip=args.clip,
        )
        val_loss = evaluate_model(model=bilstm_crf, dataloader=val_dataloader)
        end_time = time.time()

        predicted_labels = test_model(
            test_data=test,
            model=bilstm_crf,
            batch_size=args.batch_size,
            tokens_to_idx=tokens_to_idx,
            idx_to_tags=idx_to_tags,
        )

        gold_labels = []
        for label_lst in test["ner_tags"]:
            gold_labels.append([train_data.idx_to_tags[i] for i in label_lst])

        p, r, f1 = compute_entity_level_f1(
            predicted_labels=predicted_labels, gold_labels=gold_labels
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = os.path.join(args.out, "weights", f"bilstm_crf_{args.run_id}.pt")
            torch.save(bilstm_crf.state_dict(), out_path)

        epoch_mins, epoch_secs = calculate_epoch_time(start_time, end_time)
        LOG.info(f"#######################EPOCH_{epoch+1}#######################")
        LOG.info(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        LOG.info(f"\tTrain Loss: {train_loss:.3f}")
        LOG.info(f"\t Val. Loss: {val_loss:.3f}")
        LOG.info(f"\t Test F1: {f1:.3f}")
        LOG.info(f"\t Test Precision: {p:.3f}")
        LOG.info(f"\t Test Recall: {r:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for BiLSTM_CRF")
    parser.add_argument(
        "--out", help="output directory for logs", required=True, type=str
    )
    parser.add_argument(
        "--embedding-dim", help="dimension for embeddings", default=300, type=int
    )
    parser.add_argument(
        "--hidden-dim", help="dimension for hidden layer", default=512, type=int
    )
    parser.add_argument(
        "--num-layers", help="number of lstm layers", default=1, type=int
    )
    parser.add_argument(
        "--dropout", help="regularization parameter", default=0.2, type=float
    )
    parser.add_argument(
        "--batch-size", help="train/val batch size", default=64, type=int
    )
    parser.add_argument(
        "--clip", help="gradient clipping parameter", default=1, type=int
    )
    parser.add_argument("--epochs", help="number of epochs", default=5, type=int)
    parser.add_argument("--lr", help="learning rate for optimizer", type=float)
    parser.add_argument(
        "--run-id", help="id for the current run", type=int, required=True
    )
    args = parser.parse_args()
    main(args)
