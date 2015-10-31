
from nose.tools import assert_equal

from rembed import util


def test_crop_and_pad_example():
    def _run_asserts(seq, tgt_length, expected):
        example = {"seq": seq}
        left_padding = tgt_length - len(seq)
        util.crop_and_pad_example(example, left_padding, tgt_length,
                                  key="seq")
        assert_equal(example["seq"], expected)

    seqs = [
        ([1, 1, 1], 4, [0, 1, 1, 1]),
        ([1, 2, 3], 2, [2, 3])
    ]

    for seq, tgt_length, expected in seqs:
        yield _run_asserts, seq, tgt_length, expected
