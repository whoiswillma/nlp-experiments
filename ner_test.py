import unittest
from ner import extract_nets


class TestNerUtil(unittest.TestCase):
    
    def test_extract_net(self):
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

        nets = extract_nets(tokens, tags)

        expected = [
            ('Jane Villanueva', 'PER'), 
            ('United Airlines Holding', 'ORG'),
            ('Chicago', 'LOC')
        ]

        self.assertEqual(nets, expected)

        
if __name__ == '__main__':
    unittest.main()

