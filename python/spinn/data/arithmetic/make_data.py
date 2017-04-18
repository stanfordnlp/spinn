import gflags

import itertools
from tqdm import tqdm

from spinn import util
from spinn.data.util.arithmetic import ArithmeticData

from spinn.data.arithmetic.base import NUMBERS, FIXED_VOCABULARY
from spinn.data.arithmetic import load_sign_data, load_simple_data

import sys


class ArithmeticDataType(object):

    @property
    def LABELS(self):
        raise NotImplementedError

    def is_label(self, x):
        raise NotImplementedError


class SignData(ArithmeticDataType):

    LABELS = ["-", "+", "0"]

    def is_label(self, x):
        if x < 0:
            return 0
        if x > 0:
            return 1
        if x == 0:
            return 2


class SimpleData(ArithmeticDataType):

    LABELS = NUMBERS

    def is_label(self, x):
        try:
            return self.LABELS.index(x)
        except:
            return -1


if __name__ == '__main__':
    FLAGS = gflags.FLAGS
    gflags.DEFINE_integer("length", 5, "")
    gflags.DEFINE_integer("limit", 100, "")
    gflags.DEFINE_boolean("balance_length", False, "")
    gflags.DEFINE_string("exclude", None, "If not None, exclude any example that appears in this file.")
    gflags.DEFINE_enum("data_type", "simple", ["simple", "sign"], "")
    FLAGS(sys.argv)

    limit = FLAGS.limit
    length = FLAGS.length

    if FLAGS.data_type == "simple":
        data_type = SimpleData()
    elif FLAGS.data_type == "sign":
        data_type = SignData()

    exclude_dict = {str(L): set() for L in data_type.LABELS}

    if FLAGS.exclude is not None:
        if FLAGS.data_type == "simple":
            data_manager = load_simple_data
        elif FLAGS.data_type == "sign":
            data_manager = load_sign_data

        raw_exclude_data, vocabulary = data_manager.load_data(FLAGS.exclude)

        for ex in raw_exclude_data:
            exclude_dict[ex['label']].add(' '.join(ex['tokens']))

    label_size = limit // len(data_type.LABELS)
    length_size = label_size // ((length - 1) / 2)

    dataset = ArithmeticData(NUMBERS)
    generator = dataset.generate_prefix_seqs(length)

    for idx in tqdm(range(limit)):

        label = min(idx // label_size, len(data_type.LABELS) - 1)
        desired_length = min(3 + 2 * ((idx % label_size) // length_size), length)
        if idx >= label_size * len(data_type.LABELS):
            desired_length = length

        for ii, _ in enumerate(itertools.repeat(None)):
            result, seq = next(generator)

            if data_type.is_label(result) == label:
                if FLAGS.balance_length and len(seq) != desired_length:
                    continue

                check = " ".join(seq)
                check_label = str(data_type.LABELS[label])
                if check in exclude_dict.get(check_label):
                    continue
                else:
                    exclude_dict[check_label].add(check)

                print "{}\t{}".format(data_type.LABELS[label],
                    " ".join(dataset.convert_to_sexpr(seq)),
                    )
                break
