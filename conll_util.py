import numpy as np


def downsample(dataset, percent):
    num_examples = len(dataset)
    indicies = np.arange(num_examples)
    np.random.shuffle(indicies)
    indicies = indicies[:num_examples*percent//100]
    return dataset.select(indicies)


def get_label_mappings(dataset):
    labels = dataset.features['ner_tags'].feature
    label2id = {k: labels.str2int(k) for k in labels.names}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def num_labels(dataset):
    return dataset.features['ner_tags'].feature.num_classes

