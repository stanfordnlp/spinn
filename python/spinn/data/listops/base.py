from spinn import util

NUMBERS = range(10)

FIXED_VOCABULARY = {str(x): i + 1 for i, x in enumerate(NUMBERS)}
FIXED_VOCABULARY.update({
    util.PADDING_TOKEN: 0,
    "[MIN": len(FIXED_VOCABULARY) + 1,
    "[MAX": len(FIXED_VOCABULARY) + 2,
    "[FIRST": len(FIXED_VOCABULARY) + 3,
    "[LAST": len(FIXED_VOCABULARY) + 4,
    "]": len(FIXED_VOCABULARY) + 5
})
assert len(set(FIXED_VOCABULARY.values())) == len(FIXED_VOCABULARY.values())
