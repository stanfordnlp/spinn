import gflags

import itertools

from spinn import util
from spinn.data.util.arithmetic import ArithmeticData

import sys

from spinn.data.dual_arithmetic.base import NUMBERS, FIXED_VOCABULARY
from spinn.data.dual_arithmetic import load_eq_data, load_relational_data


class ArithmeticDataType(object):

    @property
    def LABELS(self):
        raise NotImplementedError

    def is_label(self, x):
        raise NotImplementedError


class EqData(ArithmeticDataType):

    LABELS = [True, False]

    def is_label(self, x):
        r1, r2 = x
        for i, b in enumerate(self.LABELS):
            if (r1 == r2) == b:
                return i


class RelationalData(ArithmeticDataType):

    LABELS = ["<", ">", "="]

    def is_label(self, x):
        r1, r2 = x
        if r1 < r2:
            return 0
        if r1 > r2:
            return 1
        if r1 == r2:
            return 2


if __name__ == '__main__':
    FLAGS = gflags.FLAGS
    gflags.DEFINE_integer("length", 5, "")
    gflags.DEFINE_integer("limit", 10, "")
    gflags.DEFINE_enum("data_type", "eq", ["eq", "relational"], "")

    # This isn't really a problem for sentence pair data.
    gflags.DEFINE_string("exclude", None, "If not None, exclude any example that appears in this file.")

    FLAGS(sys.argv)

    assert FLAGS.exclude is None, "When generating 10k examples, found 46 duplicates. This isn't necessary."

    limit = FLAGS.limit
    length = FLAGS.length

    if FLAGS.data_type == "eq":
        data_type = EqData()
    elif FLAGS.data_type == "relational":
        data_type = RelationalData()

    if FLAGS.exclude is not None:
        def hash(ex):
            return "p {} h {}".format(
                " ".join(ex["premise_tokens"]),
                " ".join(ex["hypothesis_tokens"]),
                )

        if FLAGS.data_type == "eq":
            data_manager = load_eq_data
        elif FLAGS.data_type == "relational":
            data_manager = load_relational_data

        raw_exclude_data, vocabulary = data_manager.load_data(FLAGS.exclude)

        exclude_dict = {str(L): set() for L in data_type.LABELS}

        for ex in raw_exclude_data:
            exclude_dict[ex['label']].add(hash(ex))

    label_size = limit // len(data_type.LABELS)

    dataset = ArithmeticData(NUMBERS)
    generator = dataset.generate_prefix_seqs(length)

    skip_counter = 0
    for idx in range(limit):

        label = min(idx // label_size, len(data_type.LABELS) - 1)

        for ii, _ in enumerate(itertools.repeat(None)):
            if ii % 100 == 0:
                result1, seq1 = next(generator)
            result2, seq2 = next(generator)

            if data_type.is_label((result1, result2)) == label:
                if FLAGS.exclude is not None:
                    check = hash(dict(premise_tokens=seq1, hypothesis_tokens=seq2))
                    check_label = str(data_type.LABELS[label])
                    if check in exclude_dict.get(check_label):
                        skip_counter += 1
                        continue

                print "{}\t{}\t{}".format(data_type.LABELS[label],
                    " ".join(dataset.convert_to_sexpr(seq1)),
                    " ".join(dataset.convert_to_sexpr(seq2)),
                    )
                break
    print skip_counter
