from collections import deque
import random
import copy


class ArithmeticData(object):

    def __init__(self, NUMBERS):
        self.NUMBERS = NUMBERS
        self.ops = ['+', '-']
        self.numbers = [str(x) for x in NUMBERS]
        self.all_tokens = self.ops + self.numbers


    def eval_prefix_seq(self, seq):
        token = seq.popleft()
        if token == '+':
            return self.eval_prefix_seq(seq) + self.eval_prefix_seq(seq)
        elif token == '-':
            return self.eval_prefix_seq(seq) - self.eval_prefix_seq(seq)
        else:
            return int(token)


    def gen_prefix_seq(self, max_len):
        length = random.randint(3, max_len)

        seq = [random.choice(self.ops)]
        depth = 2
        for _ in range(length - 1):
            choice = None
            if depth >= 1:
                if random.random() < 0.4:
                    choice = random.choice(self.ops)
                else:
                    choice = random.choice(self.all_tokens)

            if choice is None:
                break

            if choice in self.ops:
                depth += 1
            else:
                depth -= 1
            seq.append(choice)

        return deque(seq)


    def generate_prefix_seqs(self, max_len, min=None, max=None):
        min = self.NUMBERS[0] if min is None else min
        max = self.NUMBERS[-1] if max is None else max
        while True:
            try:
                seq = self.gen_prefix_seq(max_len)
                result = self.eval_prefix_seq(copy.copy(seq))
            except: pass
            else:
                if result >= min and result <= max:
                    yield result, seq


    def convert_to_sexpr_helper(self, seq):
        # Runs in O(n^2) time but is correct.
        op = seq[0]
        left_start = 1
        depth = 1
        right_start = None
        for i in range(left_start, len(seq)):
            if seq[i] in self.ops:
                depth += 1
            else:
                depth -= 1
            if depth == 0:
                right_start = i + 1
                break

        if right_start - left_start == 1:
            left = [seq[left_start]]
        else:
            left = self.convert_to_sexpr_helper(seq[left_start:right_start])

        if len(seq) - right_start == 1:
            right = [seq[right_start]]
        else:
            right = self.convert_to_sexpr_helper(seq[right_start:])

        return ["(", op, "("] + left + right + [")", ")"]

    def convert_to_sexpr(self, prefix_seq):
        seq = list(prefix_seq)

        return self.convert_to_sexpr_helper(seq)
