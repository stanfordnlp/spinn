"""
Utility for plotting the various costs and accuracies vs training iteration no. Reads these values from
a log file. Can also be used to compare multiple logs by supplying multiple paths.

Example Log:
17-02-11 23:06:46 [1] Step: 100 Acc: 0.38344  0.00000 Cost: 1.00246 0.99532 0.00000 0.00714 Time: 0.00024
17-02-11 23:06:47 [1] Step: 100 Eval acc: 0.486055   0.000000 ../../spinn/snli_1.0/snli_1.0_dev.jsonl Time: 0.000007

RL:
17-02-12 22:50:18 [1] Step: 0 Acc: 0.21875 0.63616 Cost: 1.83382 1.69467 0.70488 0.00000 r0.13915 Time: 0.00673

RL-Policy:
17-02-12 22:50:26 [1] Step: 0 Acc: 0.46875 0.63750 Cost: 1.69124 1.46981 0.69752 0.00000 r-0.02642 p0.24785 Time: 0.00637
"""

import gflags
import sys
import ast


FLAGS = gflags.FLAGS


class LogLine(object):
    def __init__(self, line):
        tokens = line.split()
        parts = self.get_parts()
        assert len(tokens) == len(parts)
        for t, k in zip(tokens, parts):
            if k == '_': continue
            setattr(self, k, t)

    def get_parts(self):
        raise NotImplementedError

    def __setattr__(self, key, val):
        if key == 'step':
            val = int(val)
        elif key in ('acc', 'transition_acc'):
            val = float(val)
        elif key in ('total_cost', 'xent_cost', 'transition_cost', 'l2_cost'):
            val = float(val)
        elif key == 'time_per_example':
            val = float(val)

        return super(LogLine, self).__setattr__(key, val)


class RLTrainLine(LogLine):
    def get_parts(self):
        return [
            'date', 'time', '_',
            '_','step', '_', 'acc', 'transition_acc',
            '_', 'total_cost', 'xent_cost', 'transition_cost', 'l2_cost', 'rl_cost',
            '_', 'time_per_example',
        ]

    def __setattr__(self, key, val):
        if key == 'rl_cost':
            val = float(val[1:])

        return super(RLTrainLine, self).__setattr__(key, val)


class RLPolicyTrainLine(RLTrainLine):
    def get_parts(self):
        return [
            'date', 'time', '_',
            '_','step', '_', 'acc', 'transition_acc',
            '_', 'total_cost', 'xent_cost', 'transition_cost', 'l2_cost', 'rl_cost', 'policy_cost',
            '_', 'time_per_example',
        ]

    def __setattr__(self, key, val):
        if key == 'policy_cost':
            val = float(val[1:])

        return super(RLPolicyTrainLine, self).__setattr__(key, val)


class TrainLine(LogLine):
    def get_parts(self):
        return [
            'date', 'time', '_',
            '_','step', '_', 'acc', 'transition_acc',
            '_', 'total_cost', 'xent_cost', 'transition_cost', 'l2_cost',
            '_', 'time_per_example',
        ]


class EvalLine(LogLine):
    def get_parts(self):
        return [
            'date', 'time', '_',
            '_','step', '_', '_', 'acc', 'transition_acc',
            '_', '_', 'time_per_example',
        ]


def is_rl(line):
    return is_train(line) and len(line.strip().split(' ')) == 16


def is_rl_policy(line):
    return is_train(line) and len(line.strip().split(' ')) == 17


def is_train(line):
    return 'Acc' in line


def is_eval(line):
    return 'Eval' in line


def read(log_path):
    l_train = []
    l_eval = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()

            if is_rl_policy(line):
                l_train.append(RLPolicyTrainLine(line))
            elif is_rl(line):
                l_train.append(RLTrainLine(line))
            elif is_train(line):
                l_train.append(TrainLine(line))
            elif is_eval(line):
                l_eval.append(EvalLine(line))
    return l_train, l_eval


def GetFLAGS(log_path):
    FLAGS_str = ""
    with open(log_path) as f:
        start_reading = False
        for line in f:
            if "Flag values" in line:
                start_reading = True
                continue
            if start_reading:
                FLAGS_str += line.strip()
                if line.strip().endswith("}"):
                    break
    parsed_FLAGS = ast.literal_eval(FLAGS_str)
    return parsed_FLAGS


def Filter(log):
    ret = []
    for l in reversed(log):
        if len(ret) == 0 or l.step < ret[-1].step:
            ret.append(l)
    return ret


class Failed(object):
    acc = 0.0
    step = 0


def Analyze(limit=10):
    log_paths = FLAGS.path.split(',')
    for i, lp in enumerate(log_paths):
        print("{:03}: {}".format(i, lp))
    print

    summary_keys = [
    'best_train_acc',
    'best_train_transition_acc',
    'best_train_step',
    'last_train_step',
    'last_avg_train_acc',
    'last_avg_train_transition_acc',
    'best_eval_acc',
    'best_eval_transition_acc',
    'best_eval_step',
    'last_eval_step',
    'last_avg_eval_acc',
    'last_avg_eval_transition_acc'
    ]

    summary_dict = {k: list() for k in summary_keys}

    for i, lp in enumerate(log_paths):
        l_train, l_eval = read(lp)
        l_train, l_eval = Filter(l_train), Filter(l_eval)

        # Analyze Train
        if len(l_eval) > 0:
            best_train = max(l_train, key=lambda x: x.acc)
            last_train = max(l_train, key=lambda x: x.step)
            last_train = sorted(l_train, key=lambda x: x.step)[-limit:]
            last_avg_train_acc = sum([x.acc for x in last_train]) / float(len(last_train))
            last_avg_train_transition_acc = sum([x.transition_acc for x in last_train]) / float(len(last_train))
        else:
            top_avg_train_acc = 0.0
            top_avg_train_transition_acc = 0.0
            last_train = Failed()

        # Analyze Eval
        if len(l_eval) > 0:
            best_eval = max(l_eval, key=lambda x: x.acc)
            last_eval = sorted(l_eval, key=lambda x: x.step)[-limit:]
            last_avg_eval_acc = sum([x.acc for x in last_eval]) / float(len(last_eval))
            last_avg_eval_transition_acc = sum([x.transition_acc for x in last_eval]) / float(len(last_eval))
        else:
            top_eval, last_eval = Failed(), Failed()

        print("{:03}: Train: {:.5f} {:.5f} {:6} {:6} {:.5f} {:.5f} Eval: {:.5f} {:.5f} {:6} {:6} {:.5f} {:.5f}".format(i,
            best_train.acc, best_train.transition_acc, best_train.step, last_train[-1].step, last_avg_train_acc, last_avg_train_transition_acc,
            best_eval.acc, best_eval.transition_acc, best_eval.step, last_eval[-1].step, last_avg_eval_acc, last_avg_eval_transition_acc,
            ))
        summary_dict['best_train_acc'].append(best_train.acc)
        summary_dict['best_train_transition_acc'].append(best_train.transition_acc)
        summary_dict['best_train_step'].append(best_train.step)
        summary_dict['last_train_step'].append(last_train[-1].step)
        summary_dict['last_avg_train_acc'].append(last_avg_train_acc)
        summary_dict['last_avg_train_transition_acc'].append(last_avg_train_transition_acc)
        summary_dict['best_eval_acc'].append(best_eval.acc)
        summary_dict['best_eval_transition_acc'].append(best_eval.transition_acc)
        summary_dict['best_eval_step'].append(best_eval.step)
        summary_dict['last_eval_step'].append(last_eval[-1].step)
        summary_dict['last_avg_eval_acc'].append(last_avg_eval_acc)
        summary_dict['last_avg_eval_transition_acc'].append(last_avg_eval_transition_acc)
    print

    flag_vals_0 = FLAGS.flag_vals_0.split(',')
    flag_vals_1 = FLAGS.flag_vals_1.split(',')
    print("job_id,{flags_0},{stats},{flags_1}".format(flags_0=",".join(flag_vals_0), flags_1=",".join(flag_vals_1), stats=",".join(summary_keys)))
    for i, lp in enumerate(log_paths):
        job_flags = GetFLAGS(lp)
        flags_str_0 = ",".join([str(job_flags[v]) for v in flag_vals_0])
        flags_str_1 = ",".join([str(job_flags[v]) for v in flag_vals_1])
        stats_str = ",".join([str(summary_dict[k][i]) for k in summary_keys])
        print("{:03},{flags_0},{stats},{flags_1}".format(i, stats=stats_str, flags_0=flags_str_0, flags_1=flags_str_1))


if __name__ == '__main__':

    gflags.DEFINE_string("path", None, "Path to log file")
    gflags.DEFINE_integer("limit", 10, "How many of the top items to consider.")
    gflags.DEFINE_boolean("csv", True, "List FLAGS as CSV for each log file")
    gflags.DEFINE_string("flag_vals_0", "sha,experiment_name,model_type", "FLAG values to print to csv")
    gflags.DEFINE_string("flag_vals_1", "model_dim,tracking_lstm_hidden_dim,learning_rate,l2_lambda,embedding_keep_rate,semantic_classifier_keep_rate,transition_weight", "FLAG values to print to csv")

    FLAGS(sys.argv)
    assert FLAGS.path is not None, "Must provide a log path"
    Analyze(FLAGS.limit)

