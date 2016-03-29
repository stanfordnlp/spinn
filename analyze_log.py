"""
Utility for plotting the various costs and accuracies vs training iteration no. Reads these values from
a log file. Can also be used to compare multiple logs by supplying multiple paths.
"""

import gflags
import matplotlib.pyplot as plt
import sys

FLAGS = gflags.FLAGS

class TrainLine(object):
  def __init__(self, line):
    tokens = line.split()
    self.step = int(tokens[4])
    self.pred_acc = float(tokens[6])
    self.parse_acc = float(tokens[7])
    self.total_cost = float(tokens[9])
    self.xent_cost = float(tokens[10])
    self.action_cost = float(tokens[11])
    self.l2_cost = float(tokens[12])

class EvalLine(object):
  def __init__(self, line):
    tokens = line.split()
    self.step = int(tokens[4])
    self.pred_acc = float(tokens[7])
    self.parse_acc = float(tokens[8])

class Log(object):
  def __init__(self, path):
    self.evals = []
    with open(path) as f:
      lines = filter(lambda l : 'Step' in l, f.readlines())
      # count number of eval sets
      num_logs = 1
      for i in xrange(len(lines)):
        if 'Acc' not in lines[i]:
          num_logs += 1
          self.evals.append(lines[i].strip().split()[-1])
        else:
          break
      self.corpus = [[] for _ in xrange(num_logs)]
      # read the costs and accuracies
      for i, line in enumerate(lines):
        if 'Acc' in line:
          ind = 0
        else:
          ind = self.evals.index(lines[i].strip().split()[-1]) + 1
        # ind = i % num_logs
        if  ind == 0:
          self.corpus[0].append(TrainLine(line))
        elif 'Eval' in line:
          self.corpus[ind].append(EvalLine(line))

def ShowPlots(subplot=False):
  for log_ind, path in enumerate(FLAGS.path.split(":")):
    log = Log(path)
    if subplot:
      plt.subplot(len(FLAGS.path.split(":")), 1, log_ind + 1)
    for index in FLAGS.index.split(","):
      index = int(index)
      for attr in ["pred_acc", "parse_acc", "total_cost", "xent_cost", "l2_cost", "action_cost"]:
        if getattr(FLAGS, attr):
          if "cost" in attr:
            assert index == 0, "costs only associated with training log"
          steps, val = zip(*[(l.step, getattr(l, attr)) for l in log.corpus[index] if l.step < FLAGS.iters])
          dct = {}
          for k, v in zip(steps, val):
            dct[k] = max(v, dct[k]) if k in dct else v
          steps, val = zip(*sorted(dct.iteritems()))
          plt.plot(steps, val, label="Log%d:%s-%d" % (log_ind, attr, index))
    
  plt.xlabel("No. of training iteration")
  plt.ylabel(FLAGS.ylabel)
  if FLAGS.legend:
    plt.legend()
  plt.show()


if __name__ == '__main__':
  
  gflags.DEFINE_string("path", None, "Path to log file")
  gflags.DEFINE_string("index", "1", "csv list of corpus indices. 0 for train, 1 for eval set 1 etc.")
  gflags.DEFINE_boolean("pred_acc", True, "Prediction accuracy")
  gflags.DEFINE_boolean("parse_acc", False, "Parsing accuracy")
  gflags.DEFINE_boolean("total_cost", False, "Total cost, valid only if index == 0")
  gflags.DEFINE_boolean("xent_cost", False, "Cross entropy cost, valid only if index == 0")
  gflags.DEFINE_boolean("l2_cost", False, "L2 regularization cost, valid only if index == 0")
  gflags.DEFINE_boolean("action_cost", False, "Action cost, valid only if index == 0")
  gflags.DEFINE_boolean("legend", False, "Show legend in plot")
  gflags.DEFINE_boolean("subplot", False, "Separate plots for each log")
  gflags.DEFINE_string("ylabel", "", "Label for y-axis of plot")
  gflags.DEFINE_integer("iters", 10000, "Iters to limit plot to")

  FLAGS(sys.argv)
  assert FLAGS.path is not None, "Must provide a log path"
  ShowPlots(FLAGS.subplot)
  
