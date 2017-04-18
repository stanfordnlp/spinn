from spinn import util

from spinn.data.arithmetic.base import NUMBERS, FIXED_VOCABULARY

SENTENCE_PAIR_DATA = False
OUTPUTS = ["-", "+", "0"]
LABEL_MAP = {str(x): i for i, x in enumerate(OUTPUTS)}

def load_data(path, lowercase=None):
    examples = []
    with open(path) as f:
        for example_id, line in enumerate(f):
            line = line.strip()
            label, seq = line.split('\t')
            tokens, transitions = util.convert_binary_bracketed_seq(seq.split(' '))

            example = {}
            example["label"] = label
            example["sentence"] = seq
            example["tokens"] = tokens
            example["transitions"] = transitions
            example["example_id"] = str(example_id)

            examples.append(example)
    return examples, FIXED_VOCABULARY
