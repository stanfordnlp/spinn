import unittest
import tempfile
import math
import shutil

import numpy as np

from spinn.util.metrics import size, MetricsWriter, MetricsReader


class PytorchTestCase(unittest.TestCase):

    def test_size(self):
        # We are saving a double and an integer. This should
        # fit in 12 bytes.
        assert size == 12

    def test_basic(self):
        # Save to and load from temporary dir.
        temp_dir = tempfile.mkdtemp()

        data = [(0.9, 100), (0.3, 200)]
        key = 'a'

        writer = MetricsWriter(temp_dir)
        for row in data:
            val = row[0]
            step = row[1]
            writer.write(key, val, step)

        reader = MetricsReader(temp_dir)
        table = reader.read(key)

        for r, e in zip(table, data):
            assert r[0] == e[0] and r[1] == e[1]
        
        # Cleanup temporary dir.
        shutil.rmtree(temp_dir)

    def test_offset(self):
        # Save to and load from temporary dir.
        temp_dir = tempfile.mkdtemp()

        data = [(0.9, 100), (0.3, 200)]
        key = 'a'

        writer = MetricsWriter(temp_dir)
        for row in data:
            val = row[0]
            step = row[1]
            writer.write(key, val, step)

        reader = MetricsReader(temp_dir)
        table = reader.read(key, offset=1)

        assert len(table) == 1
        assert data[1][0] == table[0][0] and data[1][1] == table[0][1]
        
        # Cleanup temporary dir.
        shutil.rmtree(temp_dir)

    def test_limit(self):
        # Save to and load from temporary dir.
        temp_dir = tempfile.mkdtemp()

        data = [(0.9, 100), (0.3, 200)]
        key = 'a'

        writer = MetricsWriter(temp_dir)
        for row in data:
            val = row[0]
            step = row[1]
            writer.write(key, val, step)

        reader = MetricsReader(temp_dir)
        table = reader.read(key, limit=1)

        assert len(table) == 1
        assert data[0][0] == table[0][0] and data[0][1] == table[0][1]
        
        # Cleanup temporary dir.
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
