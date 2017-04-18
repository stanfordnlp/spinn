"""

Really easy log parsing.

"""

try:
    from parse import *
except:
    pass
import json


FMT_TRAIN = "Train-Format: "
FMT_TRAIN_EXTRA = "Train-Extra-Format: "
FMT_EVAL = "Eval-Format: "
FMT_EVAL_EXTRA = "Eval-Extra-Format: "

IS_TRAIN = "Acc:"
IS_TRAIN_EXTRA = "Train Extra:"
IS_EVAL = "Eval acc:"
IS_EVAL_EXTRA = "Eval Extra:"

START_TRAIN = "Step:"
START_TRAIN_EXTRA = "Train Extra:"
START_EVAL = "Step:"
START_EVAL_EXTRA = "Eval Extra:"


def get_format(filename, prefix):
    with open(filename) as f:
        for line in f:
            if prefix in line:
                return line[line.find(prefix) + len(prefix):].strip()
    raise Exception("Format string not found.")


def get_json_data(filename, prefix):
    with open(filename) as f:
        for line in f:
            if prefix in line:
                data = line[line.find(prefix) + len(prefix):].strip()
                return json.loads(data)
    raise Exception("Format string not found.")


def parse_flags(filename):
    PREFIX_FLAGS = "Flag Values:\n"
    TERMINAL = "}\n"
    data = ""
    read_json = False
    with open(filename) as f:
        for line in f:
            if read_json:
                data += line
                if TERMINAL in line:
                    break
            if PREFIX_FLAGS in line:
                read_json = True
    return json.loads(data)


def is_train(line):
    return line.find(FMT_TRAIN) < 0 and line.find(IS_TRAIN) >= 0


def is_train_extra(line):
    return line.find(FMT_TRAIN_EXTRA) < 0 and line.find(IS_TRAIN_EXTRA) >= 0


def is_eval(line):
    return line.find(FMT_EVAL) < 0 and line.find(IS_EVAL) >= 0


def is_eval_extra(line):
    return line.find(FMT_EVAL_EXTRA) < 0 and line.find(IS_EVAL_EXTRA) >= 0


def read_file(filename):
    flags = parse_flags(filename)
    train_str, train_extra_str = get_format(filename, FMT_TRAIN), get_format(filename, FMT_TRAIN_EXTRA)
    eval_str, eval_extra_str = get_format(filename, FMT_EVAL), get_format(filename, FMT_EVAL_EXTRA)

    dtrain, dtrain_extra, deval, deval_extra = [], [], [], []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if is_train(line):
                dtrain.append(parse(train_str, line[line.find(START_TRAIN):].strip()))
            elif is_train_extra(line):
                dtrain_extra.append(parse(train_extra_str, line[line.find(START_TRAIN_EXTRA):].strip()))
            elif is_eval(line):
                deval.append(parse(eval_str, line[line.find(START_EVAL):].strip()))
            elif is_eval_extra(line):
                deval_extra.append(parse(eval_extra_str, line[line.find(START_EVAL_EXTRA):].strip()))

    return dtrain, dtrain_extra, deval, deval_extra, flags


if __name__ == '__main__':
    import gflags
    import sys

    FLAGS = gflags.FLAGS
    gflags.DEFINE_string("path", "scripts/sample.log", "")
    FLAGS(sys.argv)

    dtrain, dtrain_extra, deval, deval_extra, flags = read_file(FLAGS.path)

    print "Flags:"
    print "Model={model_type}\nLearning_Rate={learning_rate}".format(**flags)
    print

    print "Train:"
    for d in dtrain:
        print("Step: {} Acc: {}".format(d['step'], d['class_acc']))
    print

    print "Eval:"
    for d in deval:
        print("Step: {} Acc: {}".format(d['step'], d['class_acc']))

