import numpy as np

import os
from nose.tools import assert_equal

from spinn import util


cdir = os.path.dirname(os.path.realpath(__file__))
TEST_EMBEDDING_MATRIX = "{}/test_embedding_matrix.5d.txt".format(cdir)


def test_build_vocabulary_for_ascii_embedding_file():
    types_in_data = ["and", "the", "_", "strange_and_exotic_word"]
    core_vocabulary = {"*PADDING*":0}
    vocabulary = util.BuildVocabularyForASCIIEmbeddingFile(TEST_EMBEDDING_MATRIX, types_in_data, core_vocabulary)

    expected = {
        "*PADDING*" : 0,
        "the" : 1,
        "and" : 2,
        "_" : 3,
    }

    assert_equal(vocabulary, expected)


def test_load_embeddings_from_ascii():
    vocabulary = {"strange_and_exotic_word" : 0, "the" : 1, "." : 2}
    loaded_matrix = util.LoadEmbeddingsFromASCII(vocabulary, 5, TEST_EMBEDDING_MATRIX)
    expected = np.asarray(
        [[0, 0, 0, 0, 0],
        [0.418, 0.24968, -0.41242, 0.1217, 0.34527],
        [0.15164, 0.30177, -0.16763, 0.17684, 0.31719]], dtype=np.float32)

    np.testing.assert_array_equal(loaded_matrix, expected)

def test_crop_and_pad_example():
    def _run_asserts(seq, tgt_length, expected):
        example = {"seq": seq}
        left_padding = tgt_length - len(seq)
        util.CropAndPadExample(example, left_padding, tgt_length, "seq")
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
    # - num_transitions reports the number of transitions in the original sequence.
    expected = [
        {
            "tokens": [6, 2, 0, 0, 0],
            "transitions": [1, 0, 1, 0, 1],
            "num_transitions": 11
        },
        {
            "tokens": [0, 0, 6, 1, 0],
            "transitions": [0, 0, 0, 0, 1],
            "num_transitions": 3
        },
        {
            "tokens": [0, 0, 0, 0, 0],
            "transitions": [1, 1, 1, 1, 1],
            "num_transitions": 11
        }
    ]

    dataset = util.CropAndPad(dataset, length)
    assert_equal(dataset, expected)

if __name__ == '__main__':
    test_build_vocabulary_for_ascii_embedding_file()
    test_load_embeddings_from_ascii()
    test_crop_and_pad()
    test_crop_and_pad_example()
