"""

The old version of this script is here:
https://github.com/mrdrozdov/spinn/blob/2a06fc489faf44796dcb88a57afd699ba9937cfa/scripts/analyze_trees.py

Usage:
$ mkdir example_trees
$ python scripts/analyze_report_trees.py \
    --data_path ./snli_1.0/snli_1.0_dev.jsonl \
    --report_path ./checkpoints/example-snli.report \
    --out_path ./example_trees

"""

import gflags
import numpy as np
import pandas as pd
import sys
import pydot
import argparse
from PIL import Image
from tqdm import tqdm


from spinn.data.snli import load_snli_data


FLAGS = gflags.FLAGS


def create_tree(words, transitions):
    template_start = """
    digraph G {
        nodesep=0.4; //was 0.8
        ranksep=0.5;
    """
    template_end = """
    }
    """
    template = ""
    template += template_start
    buf = list(reversed(words))
    stack = []
    leaves = []
    for i, t in enumerate(transitions):
        if t == 0:
            stack.append((i+1,t))
            leaves.append(str(i+1))
            template += '{node[label = "%s"]; %s;}\n' % (str(buf.pop()), str(i+1))
        else:
            right = stack.pop()
            left = stack.pop()
            top = i + 1
            stack.append((top, (left, right)))
            template += "{} -> {};\n".format(top, left[0])
            template += "{} -> {};\n".format(top, right[0])
    template += "{rank=same; %s}" % ("; ".join(leaves))
    template += template_end
    return stack, template


def print_tree(words, transitions, output_file):
    _, tree = create_tree(words, transitions)
    graphs = pydot.graph_from_dot_data(tree)
    with open(output_file, 'wb') as f:
        f.write(graphs[0].create_jpeg())
    return graphs[0]


# UNUSED:
# def combine_trees(files, output_file):
#     images = map(Image.open, files)
#     widths, heights = zip(*(i.size for i in images))

#     total_width = sum(widths)
#     max_height = max(heights)

#     new_im = Image.new('L', (total_width, max_height))
#     new_im.paste(255, (0, 0, total_width, max_height))

#     x_offset = 0
#     for im in images:
#       new_im.paste(im, (x_offset,0))
#       x_offset += im.size[0]

#     new_im.save(output_file)


def read_report(fn):
    return pd.read_csv(fn, delimiter=' ', names=['example_id', 'correct', 'truth', 'pred', 'out0', 'out1', 'out2', 'sent1_parse', 'sent2_parse'])


def build_data_dict(data):
    table = dict()
    for ex in data:
        table[ex['example_id']] = ex
    return table


def run():
    print("Loading data...")
    data, _ = load_snli_data.load_data(FLAGS.data_path)
    
    print("Building data table...")
    data_table = build_data_dict(data)

    print("Loading report...")
    report = read_report(FLAGS.report_path)
    report = report.to_records()

    for rec in tqdm(report):
        ex = data_table[rec.example_id]
        transitions1 = [int(t) for t in rec.sent1_parse]
        tokens1 = data_table[rec.example_id]['premise_tokens']
        transitions2 = [int(t) for t in rec.sent2_parse]
        tokens2 = data_table[rec.example_id]['hypothesis_tokens']

        # Writes tree to file.
        graph1 = print_tree(tokens1, transitions1, "{}/{}_1.png".format(FLAGS.out_path, rec.example_id))

        # Writes tree to file.
        graph2 = print_tree(tokens2, transitions2, "{}/{}_2.png".format(FLAGS.out_path, rec.example_id))


if __name__ == '__main__':
    gflags.DEFINE_string("report_path", "./checkpoints/example-snli.report", "")
    gflags.DEFINE_string("data_path", "./snli_1.0/snli_1.0_dev.jsonl", "")
    gflags.DEFINE_string("out_path", "./example_trees", "")

    FLAGS(sys.argv)

    run()
