from collections import OrderedDict

import cPickle
import theano

from spinn.util.blocks import HeKaimingInitializer


class VariableStore(object):

    def __init__(self, prefix="vs", default_initializer=HeKaimingInitializer(), logger=None):
        self.prefix = prefix
        self.default_initializer = default_initializer
        self.vars = OrderedDict()  # Order is used in saving and loading
        self.savable_vars = OrderedDict()
        self.trainable_vars = OrderedDict()
        self.logger = logger
        self.nongradient_updates = OrderedDict()

    def add_param(self, name, shape, initializer=None, savable=True, trainable=True):
        if not initializer:
            initializer = self.default_initializer

        if name not in self.vars:
            full_name = "%s/%s" % (self.prefix, name)
            if self.logger:
                self.logger.Log(
                    "Created variable " + full_name + " shape: " + str(shape), level=self.logger.DEBUG)
            init_value = initializer(shape).astype(theano.config.floatX)
            self.vars[name] = theano.shared(init_value,
                                            name=full_name)
            if savable:
                self.savable_vars[name] = self.vars[name]
            if trainable:
                self.trainable_vars[name] = self.vars[name]

        return self.vars[name]

    def save_checkpoint(self, filename="vs_ckpt", keys=None, extra_vars=[]):
        if not keys:
            keys = self.savable_vars
        save_file = open(filename, 'w')  # this will overwrite current contents
        for key in keys:
            cPickle.dump(self.vars[key].get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
        for var in extra_vars:
            cPickle.dump(var, save_file, -1)
        save_file.close()

    def load_checkpoint(self, filename="vs_ckpt", keys=None, num_extra_vars=0, skip_saved_unsavables=False):
        if skip_saved_unsavables:
            keys = self.vars
        else:
            if not keys:
                keys = self.savable_vars
        save_file = open(filename)
        for key in keys:
            if skip_saved_unsavables and key not in self.savable_vars:
                if self.logger:
                    full_name = "%s/%s" % (self.prefix, key)
                    self.logger.Log(
                        "Not restoring variable " + full_name, level=self.logger.DEBUG)
                _ = cPickle.load(save_file) # Discard
            else:
                if self.logger:
                    full_name = "%s/%s" % (self.prefix, key)
                    self.logger.Log(
                        "Restoring variable " + full_name, level=self.logger.DEBUG)
            self.vars[key].set_value(cPickle.load(save_file), borrow=True)

        extra_vars = []
        for _ in range(num_extra_vars):
            extra_vars.append(cPickle.load(save_file))
        return extra_vars

    def add_nongradient_update(self, variable, new_value):
        # Track an update that should be applied during training but that aren't gradients.
        # self.nongradient_updates should be fed as an update to theano.function().
        self.nongradient_updates[variable] = new_value

