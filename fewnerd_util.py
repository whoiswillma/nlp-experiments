import torch


def labels_to_mappings(
    fewnerd_labels: list[str],
) -> list[int, dict[str, int], dict[int, str]]:
    num_labels = len(fewnerd_labels)
    label2id = {lbl: i for i, lbl in enumerate(fewnerd_labels)}
    id2label = {_id: lbl for lbl, _id in label2id.items()}

    return [num_labels, label2id, id2label]


def encode_fewnerd(dataset, tokenizer, label2id: dict) -> list:
    def add_encodings(example):
        encodings = tokenizer(
            example["tokens"],
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
        )
        coarse_fine_lbl: list[str] = [
            c_lbl + "-" + f_lbl for (c_lbl, f_lbl) in example["coarse_fine_labels"]
        ]
        coarse_fine_id: list[int] = [label2id[lbl] for lbl in coarse_fine_lbl]
        labels: list[int] = coarse_fine_id + [0] * (
            tokenizer.model_max_length - len(coarse_fine_id)
        )

        ret_encoding: dict[str, int] = {**encodings, "labels": labels}
        return ret_encoding

    dataset: dict[str, list[dict]] = {
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
