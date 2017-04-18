import gflags
import sys
import torch

from spinn.util.misc import recursively_set_device

FLAGS = gflags.FLAGS


def convert(inpt, outp, gpu=-1):
    ckpt = torch.load(inpt)

    if gpu < 0:
        ckpt['model_state_dict'] = {k: v.cpu() for k, v in ckpt['model_state_dict'].iteritems()}
    else:
        ckpt['model_state_dict'] = {k: v.cuda() for k, v in ckpt['model_state_dict'].iteritems()}
    ckpt['optimizer_state_dict'] = recursively_set_device(ckpt['optimizer_state_dict'], gpu)

    torch.save(ckpt, FLAGS.outp)


if __name__ == '__main__':
    gflags.DEFINE_string("inpt", None, "")
    gflags.DEFINE_string("outp", None, "")
    gflags.DEFINE_integer("gpu", -1, "")
    FLAGS(sys.argv)
    convert(FLAGS.inpt, FLAGS.outp)
