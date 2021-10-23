import unittest
from luke_util import (
    take_closure_over_entity_spans_to_labels,
    chunked,
    convert_span_labels_to_named_entity_spans,
    greedy_extract_named_entity_spans, take_closure_over_entity_spans
)


class LukeUtilTest(unittest.TestCase):

    def test_chunked(self):
        self.assertEqual(
            chunked('ABCDEFG', 3),
            ['ABC', 'DEF', 'G']
        )

        self.assertEqual(
            chunked('ABCDEF', 3),
            ['ABC', 'DEF']
        )

        self.assertEqual(
            chunked([], 1),
            []
        )

        self.assertEqual(
            chunked('A', 3),
            ['A']
        )

    def test_take_closure_over_entity_spans_to_labels(self):
        entity_spans = {
            (0, 1): 500
        }
        closure = take_closure_over_entity_spans_to_labels(entity_spans)
        expected = {
            (0, 1): 500
        }
        self.assertEqual(closure, expected)

        entity_spans = {
            (2, 4): 500
        }
        closure = take_closure_over_entity_spans_to_labels(entity_spans)
        expected = {
            (2, 3): 500,
            (3, 4): 500,
            (2, 4): 500
        }
        self.assertEqual(closure, expected)

        entity_spans = {
            (2, 4): 500, (6, 7): 501
        }
        closure = take_closure_over_entity_spans_to_labels(entity_spans)
        expected = {
            (2, 3): 500, (3, 4): 500, (2, 4): 500,
            (6, 7): 501
        }
        self.assertEqual(closure, expected)

        closure = {
            (2, 3): 500, (3, 4): 500, (2, 4): 500,
            (6, 7): 501
        }
        expected = closure
        self.assertEqual(closure, expected)


    def test_take_closure_over_entity_spans(self):
        self.assertEqual(
            {(0, 1), (0, 2), (1, 2)},
            take_closure_over_entity_spans([(0, 2)])
        )

        self.assertEqual(
            {(3, 4), (5, 6)},
            take_closure_over_entity_spans([(3, 4), (5, 6)])
        )


    def test_convert_span_labels_to_named_entity_spans(self):
        span_labels = { ((0, 1), 500), ((2, 5), 600) }
        self.assertEqual(
            { 500: [(0, 0)], 600: [(2, 4)] },
            convert_span_labels_to_named_entity_spans(span_labels)
        )


    def test_greedy_extract_named_entity_spans_selects_entity(self):
        span_label_logit = [
            ((1, 2), 500, 1), # entity span from [1, 2)
        ]
        self.assertEqual(
            { 500: [(1, 1)] },
            greedy_extract_named_entity_spans(span_label_logit, 0)
        )

        span_label_logit = [
            ((1, 2), 500, 1), # entity span from [1, 2)
            ((4, 5), 500, 1),  # entity span from [4, 5)
        ]
        self.assertEqual(
            { 500: [(1, 1), (4, 4)] },
            greedy_extract_named_entity_spans(span_label_logit, 0)
        )

        span_label_logit = [
            ((1, 2), 500, 1), # entity span from [1, 2)
            ((4, 5), 501, 1),  # entity span from [4, 5)
        ]
        self.assertEqual(
            { 500: [(1, 1)], 501: [(4, 4)] },
            greedy_extract_named_entity_spans(span_label_logit, 0)
        )


    def test_greedy_extract_named_entity_spans_ignores_nonentity(self):
        span_label_logit = [
            ((0, 5), 0, 1), # nonentity span from [0, 5)
            ((1, 2), 500, 1), # entity span from [1, 2)
            ((3, 4), 501, 2),  # entity span from [3, 4)
        ]
        self.assertEqual(
            {500: [(1, 1)], 501: [(3, 3)]},
            greedy_extract_named_entity_spans(span_label_logit, 0)
        )


    def test_greedy_extract_named_entity_spans_works_with_only_nonentity(self):
        span_label_logit = [
            ((0, 5), 0, 1), # nonentity span from [0, 5)
            ((5, 10), 0, 1),  # nonentity span from [0, 5)
        ]
        self.assertEqual(
            {},
            greedy_extract_named_entity_spans(span_label_logit, 0)
        )


    def test_greedy_extract_named_entity_spans_selects_entity_by_decreasing_logit(self):
        span_label_logit = [
            ((1, 2), 500, 1),
            ((1, 2), 501, 2), # label 501 has highest logit value
            ((1, 2), 502, 1),
        ]
        self.assertEqual(
            { 501: [(1, 1)] },
            greedy_extract_named_entity_spans(span_label_logit, 0)
        )

        span_label_logit = [
            ((1, 2), 500, 1),
            ((1, 3), 501, 2),
            ((2, 3), 502, 3),
        ]
        self.assertEqual(
            { 500: [(1, 1)], 502: [(2, 2)] },
            greedy_extract_named_entity_spans(span_label_logit, 0)
        )


if __name__ == '__main__':
    unittest.main()
