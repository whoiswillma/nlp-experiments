import unittest
from ner import (
    extract_nets_from_bio,
    extract_nets_from_chunks,
    extract_named_entity_spans_from_bio,
    compute_binary_confusion_matrix_from_bio,
    NERBinaryConfusionMatrix,
    compute_binary_confusion_matrix_from_batched_bio,
    extract_named_entity_spans_from_chunks,
)


class TestNerUtil(unittest.TestCase):
    def test_extract_nets_from_chunks(self):
        # example from https://web.stanford.edu/~jurafsky/slp3/8.pdf
        tokens_and_tags = [
            ("Jane", "PER"),
            ("Villanueva", "PER"),
            ("of", "O"),
            ("United", "ORG"),
            ("Airlines", "ORG"),
            ("Holding", "ORG"),
            ("discussed", "O"),
            ("the", "O"),
            ("Chicago", "LOC"),
            ("route", "O"),
            (".", "O"),
        ]

        tokens = [x[0] for x in tokens_and_tags]
        tags = [x[1] for x in tokens_and_tags]

        nets = extract_nets_from_chunks(tokens, tags, "O")

        expected = [
            ("Jane Villanueva", "PER"),
            ("United Airlines Holding", "ORG"),
            ("Chicago", "LOC"),
        ]

        self.assertEqual(expected, nets)

    def test_extract_nets_from_chunks_correctly_extracts_last_tag(self):
        tokens_and_tags = [
            ("We", "O"),
            ("the", "O"),
            ("People", "O"),
            ("of", "O"),
            ("the", "O"),
            ("United", "ORG"),
            ("States", "ORG"),
        ]

        tokens = [x[0] for x in tokens_and_tags]
        tags = [x[1] for x in tokens_and_tags]

        nets = extract_nets_from_chunks(tokens, tags, "O")

        expected = [("United States", "ORG")]

        self.assertEqual(expected, nets)

    def test_extract_nets_from_bio(self):
        # example from https://web.stanford.edu/~jurafsky/slp3/8.pdf
        tokens_and_tags = [
            ("Jane", "B-PER"),
            ("Villanueva", "I-PER"),
            ("of", "O"),
            ("United", "B-ORG"),
            ("Airlines", "I-ORG"),
            ("Holding", "I-ORG"),
            ("discussed", "O"),
            ("the", "O"),
            ("Chicago", "B-LOC"),
            ("route", "O"),
            (".", "O"),
        ]

        tokens = [x[0] for x in tokens_and_tags]
        tags = [x[1] for x in tokens_and_tags]

        nets = extract_nets_from_bio(tokens, tags)

        expected = [
            ("Jane Villanueva", "PER"),
            ("United Airlines Holding", "ORG"),
            ("Chicago", "LOC"),
        ]

        self.assertEqual(expected, nets)

    def test_extract_named_entity_spans_from_bio(self):
        tags = [
            "B-PER",
            "I-PER",
            "O",
            "B-ORG",
            "I-ORG",
            "I-ORG",
            "O",
            "O",
            "B-LOC",
            "O",
            "O",
        ]

        self.assertEqual(
            extract_named_entity_spans_from_bio(tags),
            {"PER": [(0, 1)], "ORG": [(3, 5)], "LOC": [(8, 8)]},
        )

    def test_extract_named_entity_spans_from_bio_touching_end(self):
        tags = [
            "B-PER",
        ]

        self.assertEqual({"PER": [(0, 0)]}, extract_named_entity_spans_from_bio(tags))

        tags = [
            "B-PER",
            "B-ORG",
        ]

        self.assertEqual(
            {"PER": [(0, 0)], "ORG": [(1, 1)]},
            extract_named_entity_spans_from_bio(tags),
        )

        tags = [
            "B-PER",
            "I-PER",
        ]

        self.assertEqual({"PER": [(0, 1)]}, extract_named_entity_spans_from_bio(tags))

        tags = [
            "B-PER",
            "I-PER",
            "B-ORG",
        ]

        self.assertEqual(
            {"PER": [(0, 1)], "ORG": [(2, 2)]},
            extract_named_entity_spans_from_bio(tags),
        )

    def test_extract_named_entity_spans_from_bio_with_multiple_entities_of_same_type(
        self,
    ):
        tags = ["B-PER", "I-PER", "O", "B-ORG", "O", "B-PER", "I-PER", "O", "B-ORG"]

        self.assertEqual(
            {"PER": [(0, 1), (5, 6)], "ORG": [(3, 3), (8, 8)]},
            extract_named_entity_spans_from_bio(tags),
        )

    def test_compute_binary_confusion_matrix_from_bio_true_positive(self):
        pred_bio = ["O", "B-PER", "O"]
        gold_bio = ["O", "B-PER", "O"]
        self.assertEqual(
            NERBinaryConfusionMatrix(tp=1),
            compute_binary_confusion_matrix_from_bio(pred_bio, gold_bio),
        )

    def test_compute_binary_confusion_matrix_from_bio_false_negative(self):
        pred_bio = ["O", "O", "O"]
        gold_bio = ["O", "B-ORG", "O"]
        self.assertEqual(
            NERBinaryConfusionMatrix(fn=1),
            compute_binary_confusion_matrix_from_bio(pred_bio, gold_bio),
        )

    def test_compute_binary_confusion_matrix_from_bio_false_positive(self):
        pred_bio = ["O", "B-PER", "O"]
        gold_bio = ["O", "O", "O"]
        self.assertEqual(
            NERBinaryConfusionMatrix(fp=1),
            compute_binary_confusion_matrix_from_bio(pred_bio, gold_bio),
        )

    def test_compute_binary_confusion_matrix_from_bio_false_positive_negative(self):
        pred_bio = ["O", "B-PER", "O"]
        gold_bio = ["O", "B-ORG", "O"]
        self.assertEqual(
            NERBinaryConfusionMatrix(fn=1, fp=1),
            compute_binary_confusion_matrix_from_bio(pred_bio, gold_bio),
        )

    def test_compute_binary_confusion_matrix_from_bio_accumulate(self):
        confusion_matrix = NERBinaryConfusionMatrix()

        pred_bio = ["O", "B-PER", "O"]
        gold_bio = ["O", "B-ORG", "O"]
        compute_binary_confusion_matrix_from_bio(pred_bio, gold_bio, confusion_matrix)
        self.assertEqual(NERBinaryConfusionMatrix(fn=1, fp=1), confusion_matrix)

        pred_bio = ["O", "B-PER", "O"]
        gold_bio = ["O", "B-ORG", "O"]
        compute_binary_confusion_matrix_from_bio(pred_bio, gold_bio, confusion_matrix)
        self.assertEqual(NERBinaryConfusionMatrix(fn=2, fp=2), confusion_matrix)

        pred_bio = ["O", "B-PER", "O"]
        gold_bio = ["O", "B-PER", "O"]
        compute_binary_confusion_matrix_from_bio(pred_bio, gold_bio, confusion_matrix)
        self.assertEqual(NERBinaryConfusionMatrix(tp=1, fn=2, fp=2), confusion_matrix)

    def test_compute_binary_confusion_matrix_from_batched_bio(self):
        pred = [["O", "B-PER", "O"], ["O", "B-PER", "O"], ["O", "B-PER", "O"]]

        gold = [["O", "B-ORG", "O"], ["O", "B-ORG", "O"], ["O", "B-PER", "O"]]

        self.assertEqual(
            NERBinaryConfusionMatrix(tp=1, fn=2, fp=2),
            compute_binary_confusion_matrix_from_batched_bio(pred, gold),
        )

    def test_extract_named_entity_spans_from_chunks(self):
        self.assertEqual(
            {"PER": [(1, 1)]}, extract_named_entity_spans_from_chunks(["O", "PER", "O"])
        )

        self.assertEqual(
            {"PER": [(0, 0)]}, extract_named_entity_spans_from_chunks(["PER"])
        )

        self.assertEqual(
            {"PER": [(0, 1)]}, extract_named_entity_spans_from_chunks(["PER", "PER"])
        )

        self.assertEqual(
            {"PER": [(0, 0)], "ORG": [(1, 1)]},
            extract_named_entity_spans_from_chunks(["PER", "ORG"]),
        )

        self.assertEqual(
            {"PER": [(0, 0)], "ORG": [(2, 2)]},
            extract_named_entity_spans_from_chunks(["PER", "O", "ORG"]),
        )

        self.assertEqual(
            {"PER": [(0, 0)], "ORG": [(2, 3)]},
            extract_named_entity_spans_from_chunks(["PER", "O", "ORG", "ORG"]),
        )

        self.assertEqual(
            {"PER": [(0, 0)], "ORG": [(2, 3)]},
            extract_named_entity_spans_from_chunks(["PER", "O", "ORG", "ORG"]),
        )

        tags = ["PER", "PER", "O", "ORG", "ORG", "ORG", "O", "O", "LOC", "O", "O"]
        self.assertEqual(
            {"PER": [(0, 1)], "ORG": [(3, 5)], "LOC": [(8, 8)]},
            extract_named_entity_spans_from_chunks(tags),
        )


if __name__ == "__main__":
    unittest.main()
