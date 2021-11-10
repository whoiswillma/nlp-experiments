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


def encode_fewnerd(dataset, tokenizer, label2id: dict) -> list:
    def add_encodings(example):
        encodings = tokenizer(
            example["tokens"],
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
        )
        coarse_fine_lbl: List[str] = [
            c_lbl + "-" + f_lbl for (c_lbl, f_lbl) in example["coarse_fine_labels"]
        ]
        coarse_fine_id: List[int] = [label2id[lbl] for lbl in coarse_fine_lbl]
        labels: List[int] = coarse_fine_id + [0] * (
            tokenizer.model_max_length - len(coarse_fine_id)
        )

        ret_encoding: Dict[str, int] = {**encodings, "labels": labels}
        return ret_encoding

    dataset: Dict[str, List[Dict]] = {
        k: list(map(add_encodings, v)) for k, v in dataset.items()
    }

    dataset_to_torch = {}
    for k, v in dataset.items():
        dataset_to_torch_list = []
        for example in v:
            example_to_torch = {
                "input_ids": torch.Tensor(example["input_ids"]).long(),
                "attention_mask": torch.Tensor(example["attention_mask"]).long(),
                "labels": torch.Tensor(example["labels"]).long(),
            }
            dataset_to_torch_list.append(example_to_torch)
        dataset_to_torch[k] = dataset_to_torch_list

    return dataset_to_torch
