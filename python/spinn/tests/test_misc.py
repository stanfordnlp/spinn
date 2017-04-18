import unittest
import tempfile
import math

from nose.plugins.attrib import attr
import numpy as np

from spinn import util
from spinn.fat_stack import SPINN

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.misc import Accumulator


class MiscTestCase(unittest.TestCase):

    def test_accumulator(self):
        A = Accumulator()

        A.add('key', 0)
        A.add('key', 0)

        assert len(A.get('key')) == 2
        assert len(A.get('key')) == 0


if __name__ == '__main__':
    unittest.main()
