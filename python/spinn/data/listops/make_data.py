import random

MIN = "[MIN"
MAX = "[MAX"
FIRST = "[FIRST"
LAST = "[LAST"
END = "]"

OPERATORS = [MIN, MAX, FIRST, LAST] 
VALUES = range(10)

VALUE_P = 0.33
MAX_ARGS = 3
MAX_DEPTH = 14

DATA_POINTS = 100000

def generate_tree(depth):
    if depth < MAX_DEPTH:
        r = random.random()
    else: 
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1))

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t

def to_string(t):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'

def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, r)
    elif r == END:  # l must be an unsaturated function.
        return l[1]
    elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
        if l[0] == MIN:
            return (l[0], min(l[1], r))
        elif l[0] == MAX:
            return (l[0], max(l[1], r))
        elif l[0] == FIRST:
            return (l[0], l[1])
        elif l[0] == LAST:
            return (l[0], r)

data = set()
while len(data) < DATA_POINTS:
    data.add(generate_tree(1))

for example in data:
    print str(to_value(example)) + '\t' + to_string(example)
