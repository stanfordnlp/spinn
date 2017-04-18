"""

A crude but effective metrics digest. Works with Visdom.

Example Usage:

$ python -m visdom.server # in a separate terminal
$ python scripts/visdom_reporter.py \
    --state ./runs/.visdom_state.json \ # keeps state for the reporter
    --root ./runs \                     # crawls directory for experiments and metric files
    --suffix .metric \                  # file extension indicating a metric file
    --verbose                           # turn on for verbose logging

"""

import gflags
from visdom import Visdom
import numpy as np
import json

from spinn.util.metrics import MetricsReader, size

import os
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [INFO] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def is_metric_file(components, suffix):
    if len(components) < 2:
        return False
    elif components[-1].endswith(suffix):
        return True
    else:
        return False


def get_components(path, suffix):
    components = path.split(os.sep)
    if not is_metric_file(components, suffix):
        return None
    else:
        experiment_name = components[-2]
        metric_name = components[-1][:-len(suffix)]
        return experiment_name, metric_name


class VisdomUpdater(object):
    def __init__(self, root, suffix, env='main', minimum=2, limit=100):
        self.env = env
        self.root = root
        self.minimum = minimum
        self.limit = limit
        self.suffix = suffix
        self.viz = Visdom()
        self.init_state()
        self.scan()

    def scan(self):
        for root, dirs, files in os.walk(self.root):

            # No need to scan files if this fails.
            canary = os.path.join(root, 'canary' + self.suffix)
            if get_components(canary, self.suffix) is None:
                continue

            for file in files:
                components = get_components(os.path.join(root, file), self.suffix)
                if components is None:
                    continue

                experiment_name, metric_name = components
                self.update_metric(experiment_name, metric_name)

    # --- State Methods ---

    def init_state(self):
        """ State maintains a dictionary:

            state: {
                [experiment_name]: {
                    [metric_name]: location in file
                }
            }
        """
        path = os.path.join(self.root, '.viz_state.json')
        if os.path.exists(path):
            self.state = json.load(open(path))
        else:
            self.state = dict()

    def get_experiment(self, experiment_name):
        return self.state.setdefault(experiment_name, dict())

    def update_state(self, experiment_name, metric_name, offset):
        experiment = self.get_experiment(experiment_name)
        experiment[metric_name] = offset
        self.save_state()

    def save_state(self):
        path = os.path.join(self.root, '.viz_state.json')
        json.dump(self.state, open(path, 'w'))

    # --- Metric Methods ---

    def read_metric(self, experiment_name, metric_name, offset=0):
        experiment_root = os.path.join(self.root, experiment_name)
        reader = MetricsReader(experiment_root)
        return reader.read(metric_name, offset=offset)

    def update_metric(self, experiment_name, metric_name):
        logger.debug("Updating: {} - {}".format(experiment_name, metric_name))
        experiment = self.get_experiment(experiment_name)
        offset = experiment.setdefault(metric_name, 0)
        table = self.read_metric(experiment_name, metric_name, offset)

        index = 0
        while True:
            if len(table) - index < self.minimum:
                break
            n = index + min(index + self.limit, len(table))

            vals, steps = zip(*table[index:n])
            steps = np.array(steps)
            vals = np.array(vals)

            success = self.viz_write(self.env, experiment_name, metric_name, steps, vals)
            if not success:
                break
            else:
                new_offset = offset + index + n
                index = index + n
                self.update_state(experiment_name, metric_name, new_offset)


    # --- Visdom Methods ---

    def viz_success(self, win):
        if win == "win does not exist":
            return False
        return True

    def viz_write(self, env, experiment_name, metric_name, steps, vals):
        viz = self.viz
        try:
            win = viz.updateTrace(X=steps, Y=vals,
                name=experiment_name, win=metric_name, env=env,
                append=True)

            if not self.viz_success(win):
                viz.line(X=steps, Y=vals,
                    win=metric_name, env=env,
                    opts={ "legend": [experiment_name], "title": metric_name })

            return True
        except Exception as e:
            import ipdb; ipdb.set_trace()
            return False


class VisdomEventHandler(FileSystemEventHandler):
    """Logs metrics to visdom."""
    def __init__(self, root, suffix, env, *args, **kwargs):
        super(VisdomEventHandler, self).__init__(*args, **kwargs)
        self.suffix = suffix
        self.updater = VisdomUpdater(root, suffix, env)

    def handle_all_events(self, event):
        if not event.is_directory:
            components = get_components(event.src_path, self.suffix)
            if components is None:
                # Not a metric file.
                return
            experiment_name, metric_name = components
            self.updater.update_metric(experiment_name, metric_name)

    # --- Overriden methods ---

    def on_moved(self, event):
        super(VisdomEventHandler, self).on_moved(event)
        # noop

    def on_created(self, event):
        super(VisdomEventHandler, self).on_created(event)
        self.handle_all_events(event)

    def on_deleted(self, event):
        super(VisdomEventHandler, self).on_deleted(event)
        # noop

    def on_modified(self, event):
        super(VisdomEventHandler, self).on_modified(event)
        self.handle_all_events(event)


if __name__ == "__main__":

    FLAGS = gflags.FLAGS
    gflags.DEFINE_string("state", ".", "Path to state. Used to update metrics.")
    gflags.DEFINE_string("root", "./runs", "Path to log directory. Will "
        "track changes to any file in this directory.")
    gflags.DEFINE_string("env", "main", "The env group for visdom.")
    gflags.DEFINE_string("suffix", ".metric", "Suffix indicating metric file.")
    gflags.DEFINE_boolean("verbose", False, "")
    FLAGS(sys.argv)

    if FLAGS.verbose:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s [DEBUG] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

    event_handler = VisdomEventHandler(FLAGS.root, FLAGS.suffix, FLAGS.env)
    observer = Observer()
    observer.schedule(event_handler, FLAGS.root, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
