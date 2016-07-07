from collections import deque
import copy
import random
import sys


NUMBERS = range(-10, 11)
OUTPUTS = range(-100, 101)
NUM_CLASSES = len(OUTPUTS)

FIXED_VOCABULARY = {str(x): x + min(NUMBERS) for x in NUMBERS}


ops = ['+', '-']
numbers = [str(x) for x in FIXED_VOCABULARY.keys()]
all_tokens = ops + numbers

def eval_prefix_seq(seq):
    token = seq.popleft()
    if token == '+':
        return eval_prefix_seq(seq) + eval_prefix_seq(seq)
    elif token == '-':
        return eval_prefix_seq(seq) - eval_prefix_seq(seq)
    else:
        return int(token)


def gen_prefix_seq(max_len):
    length = random.randint(3, max_len)

    seq = [random.choice(ops)]
    depth = 2
    for _ in range(length - 1):
        choice = None
        if depth >= 1:
            if random.random() < 0.4:
                choice = random.choice(ops)
            else:
                choice = random.choice(all_tokens)

        if choice is None:
            break

        if choice in ops:
            depth += 1
        else:
            depth -= 1
        seq.append(choice)

    return deque(seq)


def generate_prefix_seqs(max_len, min=OUTPUTS[0], max=OUTPUTS[-1]):
    while True:
        try:
            seq = gen_prefix_seq(max_len)
            result = eval_prefix_seq(copy.copy(seq))
        except: pass
        else:
            if result >= min and result <= max:
                yield result, seq


def convert_to_sexpr(prefix_seq):
    ret = []

    depth = 0
    right_branch = False
    for i in range(len(prefix_seq)):
        token = prefix_seq[i]
        if token in ops:
            ret.extend(["(", token, "("])

            depth += 2
            right_branch = False
        else:
            ret.append(token)
            if right_branch:
                ret.extend([")", ")"])
                depth -= 2
            else:
                right_branch = True

    ret.extend([")"] * depth)

    return ret


if __name__ == '__main__':
    length = 11 if len(sys.argv) < 2 else int(sys.argv[1])

    for result, seq in generate_prefix_seqs(length):
        print "%i\t%s" % (result, " ".join(convert_to_sexpr(seq)))
