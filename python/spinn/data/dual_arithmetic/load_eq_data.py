from spinn import util

from spinn.data.dual_arithmetic.base import NUMBERS, FIXED_VOCABULARY

SENTENCE_PAIR_DATA = True
OUTPUTS = [True, False]
LABEL_MAP = {str(x): i for i, x in enumerate(OUTPUTS)}

def load_data(path, lowercase=None):
    examples = []
    with open(path) as f:
        for pairID, line in enumerate(f):
            line = line.strip()
            label, s1, s2 = line.split('\t')
            tokens1, transitions1 = util.convert_binary_bracketed_seq(s1.split(' '))
            tokens2, transitions2 = util.convert_binary_bracketed_seq(s2.split(' '))

            example = {}
            example["label"] = label
            example["premise"] = s1
            example["premise_tokens"] = tokens1
            example["premise_transitions"] = transitions1
            example["hypothesis"] = s2
            example["hypothesis_tokens"] = tokens2
            example["hypothesis_transitions"] = transitions2
            example["example_id"] = str(pairID)

            examples.append(example)
    return examples, FIXED_VOCABULARY
