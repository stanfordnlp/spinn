def convert_unary_binary_bracketed_data(filename, keep_fn=lambda x: True, convert_fn=lambda x: x):
    # Build a binary tree out of a binary parse in which every
    # leaf node is wrapped as a unary constituent, as here:
    #   (4 (2 (2 The ) (2 actors ) ) (3 (4 (2 are ) (3 fantastic ) ) (2 . ) ) )
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            example = {}
            line = line.strip()
            if len(line) == 0:
                continue
            label = line[1]
            if not keep_fn(label):
                continue
            label = convert_fn(label)

            example["label"] = label
            example["sentence"] = line
            example["tokens"] = []
            example["transitions"] = []

            words = example["sentence"].replace(')', ' )')
            words = words.split(' ')

            for index, word in enumerate(words):
                if word[0] != "(":
                    if word == ")":  
                        # Ignore unary merges
                        if words[index - 1] == ")":
                            example["transitions"].append(1)
                    else:
                        # Downcase all words to match GloVe.
                        example["tokens"].append(word.lower())
                        example["transitions"].append(0)
            example["example_id"] = str(len(examples))
            examples.append(example)
    return examples