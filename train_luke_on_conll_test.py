import unittest

import train_luke_on_conll


class MyTestCase(unittest.TestCase):

    def test_get_entity_spans_to_label(self):
        labels = [0, 1, 0, 0, 1, 0, 3, 0]
        result = train_luke_on_conll.get_entity_spans_to_label(labels)
        expected = {
            (1, 2): 1,
            (4, 5): 1,
            (6, 7): 3
        }
        self.assertEqual(result, expected)

        labels = [0, 1, 1, 0, 1, 1, 3, 0]
        result = train_luke_on_conll.get_entity_spans_to_label(labels)
        expected = {
            (1, 3): 1,
            (4, 6): 1,
            (6, 7): 3
        }
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
