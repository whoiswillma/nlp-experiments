import logging
import random
#from typing import Collection, Optional, Union

import torch
from transformers import RobertaForTokenClassification, RobertaTokenizer
import conll_util

ROBERTA_VERSION = 'roberta-large'


def make_model(num_labels, id2label, label2id):
    # make luke model and tokenizer
    logging.info('Initializing Model and Tokenizer')

    model = RobertaForTokenClassification.from_pretrained(
        ROBERTA_VERSION, num_labels=num_labels)
    model.config.id2label = id2label
    model.config.label2id = label2id

    logging.info('Model initialized fresh')
    logging.info(f'model = {model}')

    return model


def make_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_VERSION)
    return tokenizer
