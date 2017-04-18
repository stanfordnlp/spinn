import unittest
import numpy as np
from collections import deque

from spinn.data.util import arithmetic


NUMBERS = range(-10, 11)


class ArithmeticTestCase(unittest.TestCase):

    def test_eval_prefix_seq_0(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['-', '-4', '-8'])
        assert dataset.eval_prefix_seq(seq) == 4

    def test_convert_to_sexpr_0(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['-', '-4', '-8'])
        expected = '( - ( -4 -8 ) )'.split(' ')
        actual = dataset.convert_to_sexpr(seq)
        assert len(expected) == len(actual)
        assert all(e == a for e, a in zip(expected, actual)), \
            "\ne: {}\na: {}".format(' '.join(expected), ' '.join(actual))

    def test_eval_prefix_seq_1(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['-', '-', '-1', '-', '-4', '-8', '5'])
        assert dataset.eval_prefix_seq(seq) == -10

    def test_convert_to_sexpr_1(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['-', '-', '-1', '-', '-4', '-8', '5'])
        expected = '( - ( ( - ( -1 ( - ( -4 -8 ) ) ) ) 5 ) )'.split(' ')
        actual = dataset.convert_to_sexpr(seq)
        assert len(expected) == len(actual)
        assert all(e == a for e, a in zip(expected, actual)), \
            "\ne: {}\na: {}".format(' '.join(expected), ' '.join(actual))

    def test_eval_prefix_seq_2(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['-', '-', '-6', '-', '-1', '-5', '0'])
        assert dataset.eval_prefix_seq(seq) == -10

    def test_convert_to_sexpr_2(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['-', '-', '-6', '-', '-1', '-5', '0'])
        expected = '( - ( ( - ( -6 ( - ( -1 -5 ) ) ) ) 0 ) )'.split(' ')
        actual = dataset.convert_to_sexpr(seq)
        assert len(expected) == len(actual)
        assert all(e == a for e, a in zip(expected, actual)), \
            "\ne: {}\na: {}".format(' '.join(expected), ' '.join(actual))

    def test_eval_prefix_seq_3(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['+', '-', '-', '-6', '-', '-1', '-5', '0', '10'])
        assert dataset.eval_prefix_seq(seq) == 0

    def test_convert_to_sexpr_3(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['+', '-', '-', '-6', '-', '-1', '-5', '0', '10'])
        expected = '( + ( ( - ( ( - ( -6 ( - ( -1 -5 ) ) ) ) 0 ) ) 10 ) )'.split(' ')
        actual = dataset.convert_to_sexpr(seq)
        assert len(expected) == len(actual)
        assert all(e == a for e, a in zip(expected, actual)), \
            "\ne: {}\na: {}".format(' '.join(expected), ' '.join(actual))

    def test_eval_prefix_seq_4(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['+', '-', '-', '-6', '-', '-1', '-5', '+', '-', '-', '-6', '-', '-1', '-5', '0', '10', '10'])
        assert dataset.eval_prefix_seq(seq) == 0

    def test_convert_to_sexpr_4(self):
        dataset = arithmetic.ArithmeticData(NUMBERS)
        seq = deque(['+', '-', '-', '-6', '-', '-1', '-5', '+', '-', '-', '-6', '-', '-1', '-5', '0', '10', '10'])
        expected = '( + ( ( - ( ( - ( -6 ( - ( -1 -5 ) ) ) ) ( + ( ( - ( ( - ( -6 ( - ( -1 -5 ) ) ) ) 0 ) ) 10 ) ) ) ) 10 ) )'.split(' ')
        actual = dataset.convert_to_sexpr(seq)
        assert len(expected) == len(actual)
        assert all(e == a for e, a in zip(expected, actual)), \
            "\ne: {}\na: {}".format(' '.join(expected), ' '.join(actual))
    

if __name__ == '__main__':
    unittest.main()
