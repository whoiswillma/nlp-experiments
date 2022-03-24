import unittest
import roberta_util

from fewnerd_util import labels_to_mappings, encode_fewnerd


class TrainFewnerdUtilTestCase(unittest.TestCase):
    def test_labels_to_mappings(self):
        labels = ["a", "b", "c", "d"]
        result = labels_to_mappings(labels)
        expected = [
            4,
            {"a": 0, "b": 1, "c": 2, "d": 3},
            {0: "a", 1: "b", 2: "c", 3: "d"},
        ]
        self.assertEqual(result, expected)

        labels = ["e"]
        result = labels_to_mappings(labels)
        expected = [1, {"e": 0}, {0: "e"}]
        self.assertEqual(result, expected)

    def test_encode_fewnerd(self):
        tokenizer = roberta_util.make_tokenizer()
        dataset = {
            "train": [
                {
                    "id": 0,
                    "tokens": ["0"],
                    "coarse_labels": ["a"],
                    "fine_labels": ["a"],
                    "coarse_fine_labels": [("a", "a")],
                }
            ]
        }
        expected_input_ids = [
            0,
            321,
            2,
        ]
        expected_attention_mask = [
            1,
            1,
            1,
        ]
        expected_labels = [1]
        for _ in range(512 - len(expected_labels)):
            expected_labels.append(0)
        for _ in range(512 - len(expected_attention_mask)):
            expected_attention_mask.append(0)
        for _ in range(512 - len(expected_input_ids)):
            expected_input_ids.append(1)

        result_input_ids = encode_fewnerd(dataset, tokenizer, {"a-a": 1})["train"][0][
            "input_ids"
        ]
        result_attention_mask = encode_fewnerd(dataset, tokenizer, {"a-a": 1})["train"][
            0
        ]["attention_mask"]
        result_labels = encode_fewnerd(dataset, tokenizer, {"a-a": 1})["train"][0][
            "labels"
        ]

        self.assertEqual(list(result_input_ids), expected_input_ids)
        self.assertEqual(list(result_attention_mask), expected_attention_mask)
        self.assertEqual(list(result_labels), expected_labels)


if __name__ == "__main__":
    unittest.main()
