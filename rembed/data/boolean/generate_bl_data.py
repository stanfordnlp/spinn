#!/usr/bin/env python

# Recursively enumerates sentences of Boolean logic with their truth values and
# writes them to disjoint training, dev, and test set files, formatted for
# import_binary_bracketed_data.py.
#
# The maximum length of the examples and the number of generated examples are both
# governed by the recursion depth.

import copy
import random

RECURSION_DEPTH = 3

TRAIN_PORTION = 0.98
DEV_PORTION = 0.01


def get_value_for_tree(tree):
    if isinstance(tree, tuple):
        if tree[0] == 'not':
            child = get_value_for_tree(tree[1])
            return not child
        else:
            left = get_value_for_tree(tree[0])
            right = get_value_for_tree(tree[2])
            if tree[1] == "and":
                return left and right
            elif tree[1] == "or":
                return left or right
            else:
                print 'syntax error', tree
    else:
        return tree


def expand(statements):
    result = copy.copy(statements)
    for statement in statements:
        result.append(('not', statement))
        for inner_statement in statements:
            result.append((statement, 'and', inner_statement))
            result.append((statement, 'or', inner_statement))
    return result


def uniq(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x):
            return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def to_string(expr):
    if isinstance(expr, int):
        return value_names[expr]
    if isinstance(expr, str):
        return expr
    elif len(expr) == 3:
        return "( " + to_string(expr[0]) + " ( " + to_string(expr[1]) + " " + to_string(expr[2]) + " ) )"
    else:
        return "( " + to_string(expr[0]) + " " + to_string(expr[1]) + " )"


if __name__ == "__main__":
    values = [0, 1]
    value_names = ['F', 'T']

    total = 0
    statements = [0, 1]
    for i in range(RECURSION_DEPTH):
        statements = expand(statements)
        statements = uniq(statements)

    outputs = []

    for i, statement in enumerate(statements):
        tv = get_value_for_tree(statement)
        tv_string = value_names[tv]

        total += 1
        outputs.append(tv_string + "\t" + to_string(statement))

    outputs = uniq(outputs)
    random.shuffle(outputs)

    filename = 'bl_train.tsv'
    f = open(filename, 'w')
    for i in range(int(TRAIN_PORTION * len(outputs))):
        output = outputs[i]
        f.write(output + "\n")
    f.close()

    filename = 'bl_dev.tsv'
    f = open(filename, 'w')
    for i in range(int(TRAIN_PORTION * len(outputs)), int((TRAIN_PORTION + DEV_PORTION) * len(outputs))):
        output = outputs[i]
        f.write(output + "\n")
    f.close()

    filename = 'bl_test.tsv'
    f = open(filename, 'w')
    for i in range(int((TRAIN_PORTION + DEV_PORTION) * len(outputs)), len(outputs)):
        output = outputs[i]
        f.write(output + "\n")
    f.close()
