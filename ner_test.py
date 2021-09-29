import unittest
from ner import extract_nets_from_bio, extract_nets_from_chunks


class TestNerUtil(unittest.TestCase):

    def test_extract_nets_from_chunks(self):
        # example from https://web.stanford.edu/~jurafsky/slp3/8.pdf
        tokens_and_tags = [
            ('Jane', 'PER'),
            ('Villanueva', 'PER'),
            ('of', 'O'),
            ('United', 'ORG'),
            ('Airlines', 'ORG'),
            ('Holding', 'ORG'),
            ('discussed', 'O'),
            ('the', 'O'),
            ('Chicago', 'LOC'),
            ('route', 'O'),
            ('.', 'O')
        ]

        tokens = [x[0] for x in tokens_and_tags]
        tags = [x[1] for x in tokens_and_tags]

        nets = extract_nets_from_chunks(tokens, tags, 'O')

        expected = [
            ('Jane Villanueva', 'PER'),
            ('United Airlines Holding', 'ORG'),
            ('Chicago', 'LOC')
        ]

        self.assertEqual(nets, expected)


    def test_extract_nets_from_chunks_correctly_extracts_last_tag(self):
        tokens_and_tags = [
            ('We', 'O'),
            ('the', 'O'),
            ('People', 'O'),
            ('of', 'O'),
            ('the', 'O'),
            ('United', 'ORG'),
            ('States', 'ORG'),
        ]

        tokens = [x[0] for x in tokens_and_tags]
        tags = [x[1] for x in tokens_and_tags]

        nets = extract_nets_from_chunks(tokens, tags, 'O')

        expected = [
            ('United States', 'ORG')
        ]

        self.assertEqual(nets, expected)

    
    def test_extract_nets_from_bio(self):
        # example from https://web.stanford.edu/~jurafsky/slp3/8.pdf
        tokens_and_tags = [
            ('Jane', 'B-PER'),
            ('Villanueva', 'I-PER'),
            ('of', 'O'),
            ('United', 'B-ORG'),
            ('Airlines', 'I-ORG'),
            ('Holding', 'I-ORG'),
            ('discussed', 'O'),
            ('the', 'O'),
            ('Chicago', 'B-LOC'),
            ('route', 'O'),
            ('.', 'O')
        ]

        tokens = [x[0] for x in tokens_and_tags]
        tags = [x[1] for x in tokens_and_tags]

        nets = extract_nets_from_bio(tokens, tags)

        expected = [
            ('Jane Villanueva', 'PER'), 
            ('United Airlines Holding', 'ORG'),
            ('Chicago', 'LOC')
        ]

        self.assertEqual(nets, expected)

        
if __name__ == '__main__':
    unittest.main()

