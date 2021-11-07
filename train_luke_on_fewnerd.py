import logging

import torch

import luke_util
import util
from fewnerdparse.dataset import FEWNERD_SUPERVISED, FEWNERD_COARSE_FINE_TYPES


def get_entity_spans_to_label(example) -> dict[tuple[int, int] : int]:
    entity_spans_to_labels: dict[tuple[int, int] : int] = {}

    outside_label = len(FEWNERD_COARSE_FINE_TYPES)

    tokens = example["tokens"]
    label_ids: list[int] = [
        FEWNERD_COARSE_FINE_TYPES.index((coarse, fine))
        if coarse and fine
        else outside_label
        for coarse, fine in zip(example["coarse_labels"], example["fine_labels"])
    ]

    current_entity_start = -1
    current_label = outside_label

    for i, (token, label) in enumerate(
        list(zip(tokens, label_ids)) + [("", outside_label)]
    ):
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

    model, tokenizer = luke_util.make_model_and_tokenizer(
        len(FEWNERD_COARSE_FINE_TYPES) + 1
    )

    NONENTITY_LABEL = len(FEWNERD_COARSE_FINE_TYPES)
    NUM_EPOCHS = 5

    # lr from LUKE paper
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    logging.debug(f"opt = {opt}")

    FEWNERD_TRAIN = FEWNERD_SUPERVISED["train"][:10]

    for epoch in util.mytqdm(range(NUM_EPOCHS)):
        stats = luke_util.make_train_stats_dict()

        for example in util.mytqdm(FEWNERD_TRAIN, desc="train"):
            opt.zero_grad()

            entity_spans_to_labels = get_entity_spans_to_label(example)

            luke_util.train_luke_model(
                model,
                tokenizer,
                example["tokens"],
                entity_spans_to_labels,
                nonentity_label=NONENTITY_LABEL,
                stats=stats,
            )

            opt.step()

        logging.info(f"stats = {stats}")
        # util.save_checkpoint(model, opt, epoch)

        # validate
        correct = 0
        total = 0

        for example in util.mytqdm(FEWNERD_TRAIN, desc="validate"):
            entity_spans_to_labels = get_entity_spans_to_label(example)

            doc_correct, doc_total = luke_util.acid_test_luke_model(
                model,
                tokenizer,
                example["tokens"],
                entity_spans_to_labels=entity_spans_to_labels,
                nonentity_label=NONENTITY_LABEL,
            )

            correct += doc_correct
            total += doc_total

        logging.info("Validation")
        logging.info(f"num_correct = {correct}")
        logging.info(f"total_predictions = {total}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.warning(e)
        raise e
