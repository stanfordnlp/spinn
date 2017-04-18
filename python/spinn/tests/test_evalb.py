import unittest


from spinn.util.evalb import bracketing, crossing


class EvalbTestCase(unittest.TestCase):

    def test_bracketing(self):
        cases = [
            ("00101", [(0,2), (0,3)]),
            ("00011", [(1,3), (0,3)]),
            ("0010101", [(0,2), (0,3), (0,4)]),
            ("0010011", [(0,2), (2,4), (0,4)]),
        ]
        for ts, exp in cases:
            ts = map(int, ts)
            tsplits = bracketing(ts)
            assert all(t == e for t, e in zip(tsplits, exp)), "{} != {}".format(tsplits, exp)

    def test_crossing(self):
        cases = [
            (map(int, "00101"), map(int, "00101"), 0),
            (map(int, "00101"), map(int, "00011"), 1),
            (map(int, "00011"), map(int, "00101"), 1),
            (map(int, "0010101"), map(int, "0010011"), 1),
            (map(int, "0010011"), map(int, "0010101"), 1),
        ]
        for i, (gold, pred, exp) in enumerate(cases):
            crosses, count = crossing(gold, pred)
            assert count == exp, "{}. {} != {}. {}".format(i, count, exp, crosses)


if __name__ == '__main__':
    unittest.main()
