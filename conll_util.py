import numpy as np


def downsample(dataset, percent):
    num_examples = len(dataset)
    indicies = np.arange(num_examples)
    np.random.shuffle(indicies)
    indicies = indicies[: num_examples * percent // 100]
    return dataset.select(indicies)


def get_label_mappings(dataset):
    labels = dataset.features["ner_tags"].feature
    label2id = {k: labels.str2int(k) for k in labels.names}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def num_labels(dataset):
    return dataset.features["ner_tags"].feature.num_classes


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
        # return the encodings and the extended ner_tags
        return {**encodings, "labels": labels}

    dataset = dataset.map(add_encodings)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset
