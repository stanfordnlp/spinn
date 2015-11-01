
from nose.tools import assert_equal

from rembed import util


def test_crop_and_pad_example():
    def _run_asserts(seq, tgt_length, expected):
        example = {"seq": seq}
        left_padding = tgt_length - len(seq)
        util.crop_and_pad_example(example, left_padding, key="seq")
        assert_equal(example["seq"], expected)

    seqs = [
        ([1, 1, 1], 4, [0, 1, 1, 1]),
        ([1, 2, 3], 2, [2, 3])
    ]

    for seq, tgt_length, expected in seqs:
        yield _run_asserts, seq, tgt_length, expected


def test_crop_and_pad():
    dataset = [
        {
            # Transitions too long -- will need to crop both
            "tokens": [1, 2, 4, 3, 6, 2],
            "transitions": [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
        },
        {
            # Transitions too short -- will need to pad transition seq and
            # adjust tokens with dummy elements accordingly
            "tokens": [6, 1],
            "transitions": [0, 0, 1]
        },
        {
            # Transitions too long; lots of pushes
            "tokens": [6, 1, 2, 3, 5, 1],
            "transitions": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        }
    ]

    length = 5

    # Expectations:
    # - When the transition sequence is too short, it will be left padded with
    #   zeros, and the corresponding token sequence will be left padded with the
    #   same number of zeros.
    # - When the transition sequence is too long, it will be cropped from the left,
    #   since this ensures that there will be a unique stack top element at the
    #   final step. If this is not the case, than most of the model's effort
    #   will go into building embeddings that stay higher in the stack, and is thus
    #   wasted. The corresponding token sequence will be cropped by removing
    #   as many elements on the left side as there were zeros removed from the
    #   transition sequence.
    expected = [
        {
            "tokens": [6, 2, 0, 0, 0],
            "transitions": [1, 0, 1, 0, 1]
        },
        {
            "tokens": [0, 0, 6, 1, 0],
            "transitions": [0, 0, 0, 0, 1]
        },
        {
            "tokens": [0, 0, 0, 0, 0],
            "transitions": [1, 1, 1, 1, 1]
        }
    ]

    dataset = util.crop_and_pad(dataset, length)
    assert_equal(dataset, expected)

if __name__ == '__main__':
    test_crop_and_pad()
    test_crop_and_pad_example()
