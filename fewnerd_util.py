def labels_to_mappings(
    fewnerd_labels: list[str],
) -> list[int, dict[str, int], dict[int, str]]:
    num_labels = len(fewnerd_labels)
    label2id = {lbl: i for i, lbl in enumerate(fewnerd_labels)}
    id2label = {_id: lbl for lbl, _id in label2id.items()}

    return [num_labels, label2id, id2label]
