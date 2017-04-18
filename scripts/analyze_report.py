"""

Investigate common correct examples to estimate potential
upside of using an ensemble.

Example output:

$ python ../scripts/analyze_report.py --path snli-16-back-cpu-1-eval.report,snli-17-enc-parse-0-eval.report,snli-16-parse-eval.report,snli-17-enc-back-0-eval.report,snli-17-enc-parse-0-eval.report --ensemble

mid correct   score: filename
  0    7921 0.83230: snli-16-back-cpu-1-eval.report
  1    7978 0.83829: snli-17-enc-parse-0-eval.report
  2    7908 0.83093: snli-16-parse-eval.report
  3    7929 0.83314: snli-17-enc-back-0-eval.report
  4    7978 0.83829: snli-17-enc-parse-0-eval.report

 m1  m2: common upside    c+u   score
  0   1:   7370   1159   8529 0.89619
  0   2:   7319   1191   8510 0.89419
  0   3:   7331   1188   8519 0.89514
  0   4:   7370   1159   8529 0.89619
  1   2:   7345   1196   8541 0.89745
  1   3:   7382   1143   8525 0.89577
  1   4:   7978      0   7978 0.83829
  2   3:   7308   1221   8529 0.89619
  2   4:   7345   1196   8541 0.89745
  3   4:   7382   1143   8525 0.89577

    all:   6696   2165   8861 0.93107

Best:     7978 0.83829 1
Total:    9517
Scores:
 control: 0.832300094568
 mean: 0.859514552905
 max: 0.854260796469
 vote: 0.856572449301 0.856152148786 (93, 0.86077545445)

NOTE: Using two+ of the same model reduces the upside of majority vote.

"""
import gflags
import numpy as np
import pandas as pd
import sys

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

FLAGS = gflags.FLAGS


def Read(fn):
    return pd.read_csv(fn, delimiter=' ', names=['example_id', 'correct', 'truth', 'pred', 'out0', 'out1', 'out2', 'sent1_parse', 'sent2_parse'])


def Compare(reports):
    assert len(reports) > 1
    assert all(r1.shape == r2.shape for r1, r2 in zip(reports[:-1], reports[1:]))
    filters = []
    for r1 in reports:
        filters.append(r1[r1.correct == True])
    intersect = np.zeros(reports[0].shape[0])
    for f1 in filters:
        for i in f1.index:
            intersect[i] += 1
    common_correct = (intersect == len(reports)).sum()
    upside = (intersect > 0).sum() - common_correct

    return common_correct, upside


def Analyze():
    report_files = FLAGS.path.split(',')

    # Read CSV reports
    reports = []
    for i, fn in enumerate(report_files):
        rep = Read(fn)
        rep = rep.sort('example_id')
        rep.id = i
        reports.append(rep)

    # Compare Every Pair
    common = []
    for i1, rep1 in enumerate(reports):
        for i2, rep2 in enumerate(reports):
            if i2 > i1:
                common_correct, upside = Compare([rep1, rep2])
                common.append((i1, i2, common_correct, upside))

    # Compare Whole
    all_common_correct, all_upside = Compare(reports)

    # Print Report
    total = reports[0].shape[0]
    print("{:>3} {:>6} {:>7}: {}".format("mid", "correct", "score", "filename"))
    for i, (fn, report) in enumerate(zip(report_files, reports)):
        print("{:3} {:7} {:6.5f}: {}".format(i, report.correct.sum(), report.correct.sum() / float(total), fn))
    print

    best = max(reports, key=lambda x: x.correct.sum())
    print("{:>3} {:>3}: {:>6} {:>6} {:>6} {:>7}".format("m1", "m2", "common", "upside", "c+u", "score"))
    for i1, i2, c, u in common:
        print("{:3} {:3}: {:6} {:6} {:6} {:6.5f}".format(i1, i2, c, u, c + u, (c+u)/float(total)))
    print
    print("    all: {:6} {:6} {:6} {:6.5f}".format(all_common_correct, all_upside, all_common_correct + all_upside, (all_common_correct + all_upside)/float(total)))
    print
    print("Best:\t{:6} {:6.5f} {}".format(best.correct.sum(), best.correct.sum()/float(total), best.id))
    print("Total:\t{:6}".format(total))

    if FLAGS.ensemble:
        ensemble = list()

        scores = []

        # control
        control_idx = len(scores)
        scores.append(0)

        # mean
        mean_idx = len(scores)
        scores.append(0)

        # max
        max_idx = len(scores)
        scores.append(0)

        # vote simple
        vote_simple_idx = len(scores)
        scores.append(0)

        # vote weighted
        vote_weighted_idx = len(scores)
        scores.append(0)

        # vote ideal
        vote_ideal_idx = len(scores)
        scores.append(0)

        ties = 0

        for i, rs in enumerate(zip(*[r.to_records() for r in reports])):
            m_row = []
            for r_row in rs:
                m_row.append([r_row.out0, r_row.out1, r_row.out2])

            truth = r_row.truth

            m_dist = np.array(m_row).astype(np.float32)

            # Softmax as Batch
            m_dist = F.softmax(Variable(torch.from_numpy(m_dist)))

            # control
            control_guess = m_dist[0:1].max(1)[1].data[0][0]

            # mean
            mean_guess = m_dist.mean(0).max(1)[1].data[0][0]

            # max
            max_guess = m_dist.max(0)[0].max(1)[1].data[0][0]

            # vote
            num_labels = 3
            majority = m_dist.size(0) // 2 + 1
            votes = m_dist.max(1)[1].data
            vote_count = np.array([(votes == label).sum() for label in range(num_labels)])
            winners = np.arange(num_labels)[vote_count >= majority]

            if winners.shape[0] >= 1:
                assert winners.shape[0] == 1
                vote_simple_guess = winners[0]
                vote_weighted_guess = vote_simple_guess
                vote_ideal_guess = vote_simple_guess
            else: # tie
                vote_simple_guess = mean_guess

                # TODO: Use more clever weighted voting.
                vote_weights = np.zeros(num_labels)
                for model_idx, v in enumerate(votes.view(-1)):
                    vote_weights[v] += m_dist.data[model_idx, v]
                vote_weighted_guess = vote_weights.argmax()

                vote_ideal_guess = truth
                ties += 1

            scores[control_idx] += truth == control_guess
            scores[mean_idx] += truth == mean_guess
            scores[max_idx] += truth == max_guess
            scores[vote_simple_idx] += truth == vote_simple_guess
            scores[vote_weighted_idx] += truth == vote_weighted_guess
            scores[vote_ideal_idx] += truth == vote_ideal_guess

        output = "Scores:"
        output += "\n control: {control}"
        output += "\n mean: {mean}"
        output += "\n max: {max}"
        output += "\n vote: {vote_simple} {vote_weighted} ({ties}, {vote_ideal})"
        print(output.format(
            control=scores[control_idx]/float(total),
            mean=scores[mean_idx]/float(total),
            max=scores[max_idx]/float(total),
            vote_simple=scores[vote_simple_idx]/float(total),
            vote_weighted=scores[vote_weighted_idx]/float(total),
            ties=ties,
            vote_ideal=(scores[vote_ideal_idx])/float(total)
            ))


if __name__ == '__main__':

    gflags.DEFINE_string("path", None, "List of report files.")
    gflags.DEFINE_boolean("ensemble", False, "Ensemble outputs.")

    FLAGS(sys.argv)
    assert FLAGS.path is not None, "Must provide a report path"
    Analyze()

