# coding: utf-8

"""
Sparklines in ascii
source: https://github.com/rory/ascii_sparks/blob/master/ascii_sparks.py

"""

parts = u' ▁▂▃▄▅▆▇▉'

def sparks(nums):
    fraction = max(nums) / float(len(parts) - 1)
    return ''.join(parts[int(round(x/fraction))] for x in nums)
