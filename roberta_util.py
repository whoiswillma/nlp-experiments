import logging
from typing import List, Set, Dict, Tuple, Optional
from transformers import RobertaForTokenClassification, RobertaTokenizer
import torch

ROBERTA_VERSION = "roberta-base"


def make_model(num_labels, id2label, label2id):
    # make luke model and tokenizer
    logging.info("Initializing Model and Tokenizer")

    model = RobertaForTokenClassification.from_pretrained(
        ROBERTA_VERSION, num_labels=num_labels
    )
    model.config.id2label = id2label
    model.config.label2id = label2id

    logging.info("Model initialized fresh")
    logging.info(f"model = {model}")

    return model


def make_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_VERSION)
    return tokenizer


def encode_conll(dataset, tokenizer):
    def add_encodings(example):

        # get the encodings of the tokens. The tokens are already split, that is why we must add is_split_into_words=True
        encodings = tokenizer(
            example["tokens"],
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
        )
        # extend the ner_tags so that it matches the max_length of the input_ids
        labels = example["ner_tags"] + [0] * (
            tokenizer.model_max_length - len(example["ner_tags"])
        )
        return {**encodings, "labels": labels}

    dataset = dataset.map(add_encodings)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=5)
    print(next(iter(dataloader)), type(next(iter(dataloader))))
    return dataset
