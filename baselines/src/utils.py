import torch
import torch.nn as nn
from scipy.stats import mstats
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
from os.path import dirname, abspath, exists, join, isfile, expanduser
from os import makedirs, system, environ, remove
from socket import gethostname
from collections import OrderedDict
import klepto
import subprocess
from threading import Timer
from time import time
import datetime, pytz
import re
import requests
import random
import pickle
import signal
import numpy as np
import scipy.sparse as sp
import scipy
import sys
from pathlib import Path


def check_nx_version():
    # nxvg = '2.2'
    nxva = nx.__version__
    if not nxva.startswith('2'):
        # if nxvg != nxva:
        raise RuntimeError(
            f'Wrong networkx version! Need >=2 instead of {nxva}')


# Always check the version first.
check_nx_version()


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_save_path():
    return join(get_root_path(), 'save')


def get_src_path():
    return join(get_root_path(), 'src')


def create_dir_if_not_exists(dir):
    if not exists(dir):
        makedirs(dir)


def _get_y(data):
    return getattr(data, FLAGS.target.replace('-', '_'))


def _get_y_with_target(data, target):
    return getattr(data, target.replace('-', '_'))


def _get_y_multi_obj(data):
    assert (isinstance(FLAGS.target, list))
    y_list = [getattr(data, t.replace('-', '_')) for t in FLAGS.target]
    return y_list


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def save(obj, filepath, print_msg=True, use_klepto=True):
    if type(obj) is not dict and type(obj) is not OrderedDict:
        raise ValueError('Can only save a dict or OrderedDict'
                         ' NOT {}'.format(type(obj)))
    fp = proc_filepath(filepath, ext='.klepto' if use_klepto else '.pickle')
    if use_klepto:
        create_dir_if_not_exists(dirname(filepath))
        save_klepto(obj, fp, print_msg)
    else:
        save_pickle(obj, fp, print_msg)


def load(filepath, print_msg=True):
    fp = proc_filepath(filepath)
    if isfile(fp):
        return load_klepto(fp, print_msg)
    elif print_msg:
        print('Trying to load but no file {}'.format(fp))


def save_klepto(dic, filepath, print_msg):
    if print_msg:
        print('Saving to {}'.format(filepath))
    klepto.archives.file_archive(filepath, dict=dic).dump()


def load_klepto(filepath, print_msg):
    rtn = klepto.archives.file_archive(filepath)
    rtn.load()
    if print_msg:
        print('Loaded from {}'.format(filepath))
    return rtn


def save_pickle(dic, filepath, print_msg=True, protocal=4):  # protocal shouldn't be too high to avoid loading issue
    if print_msg:
        print('Saving to {}'.format(filepath))
    with open(filepath, 'wb') as handle:
        if sys.version_info.major < 3:  # python 2
            pickle.dump(dic, handle)
        elif sys.version_info >= (3, 4):  # qilin & feilong --> 3.4
            pickle.dump(dic, handle, protocol=protocal)
        else:
            raise NotImplementedError()


def load_pickle(filepath, print_msg=True):
    if isfile(filepath):
        with open(filepath, 'rb') as handle:
            pickle_data = pickle.load(handle)
            if print_msg:
                print(f'Loaded pickle from {filepath}')
            return pickle_data
    elif print_msg:
        print('No file {}'.format(filepath))


def proc_filepath(filepath, ext='.klepto'):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    return append_ext_to_filepath(ext, filepath)


def append_ext_to_filepath(ext, fp):
    if not fp.endswith(ext):
        fp += ext
    return fp


def sorted_nicely(l, reverse=False):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn


global_exec_print = True


def exec_turnoff_print():
    global global_exec_print
    global_exec_print = False


def exec_turnon_print():
    global global_exec_print
    global_exec_print = True


def global_turnoff_print():
    import sys, os
    sys.stdout = open(os.devnull, 'w')


def global_turnon_print():
    import sys
    sys.stdout = sys.__stdout__


def exec_cmd(cmd, timeout=None, exec_print=True):
    '''
    TODO: take a look at

        def _run_prog(self, prog='nop', args=''):
        """Apply graphviz program to graph and return the result as a string.

        >>> A = AGraph()
        >>> s = A._run_prog() # doctest: +SKIP
        >>> s = A._run_prog(prog='acyclic') # doctest: +SKIP

        Use keyword args to add additional arguments to graphviz programs.
        """
        runprog = r'"%s"' % self._get_prog(prog)
        cmd = ' '.join([runprog, args])
        dotargs = shlex.split(cmd)
        p = subprocess.Popen(dotargs,
                             shell=False,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             close_fds=False)
        (child_stdin,
         child_stdout,
         child_stderr) = (p.stdin, p.stdout, p.stderr)
        # Use threading to avoid blocking
        data = []
        errors = []
        threads = [PipeReader(data, child_stdout),
                   PipeReader(errors, child_stderr)]
        for t in threads:
            t.start()

        self.write(child_stdin)
        child_stdin.close()

        for t in threads:
            t.join()
        p.wait()

        if not data:
            raise IOError(b"".join(errors).decode(self.encoding))

        if len(errors) > 0:
            warnings.warn(b"".join(errors).decode(self.encoding), RuntimeWarning)
        return b"".join(data)

        taken from /home/xxx/.local/lib/python3.7/site-packages/pygraphviz/agraph.py
    '''
    global global_exec_print
    if not timeout:
        if global_exec_print and exec_print:
            print(cmd)
        else:
            cmd += ' > /dev/null'
        system(cmd)
        return True  # finished
    else:
        def kill_proc(proc, timeout_dict):
            timeout_dict["value"] = True
            proc.kill()

        def run(cmd, timeout_sec):
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            timeout_dict = {"value": False}
            timer = Timer(timeout_sec, kill_proc, [proc, timeout_dict])
            timer.start()
            stdout, stderr = proc.communicate()
            timer.cancel()
            return proc.returncode, stdout.decode("utf-8"), \
                   stderr.decode("utf-8"), timeout_dict["value"]

        if global_exec_print and exec_print:
            print('Timed cmd {} sec(s) {}'.format(timeout, cmd))
        _, _, _, timeout_happened = run(cmd, timeout)
        if global_exec_print and exec_print:
            print('timeout_happened?', timeout_happened)
        return not timeout_happened


tstamp = None


def get_ts():
    global tstamp
    if not tstamp:
        tstamp = get_current_ts()
    return tstamp


def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime(
        '%Y-%m-%dT%H-%M-%S.%f')


class timeout:
    """
    https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def get_user():
    try:
        home_user = expanduser("~").split('/')[-1]
    except:
        home_user = 'user'
    return home_user


def get_host():
    host = environ.get('HOSTNAME')
    if host is not None:
        return host
    rtn = gethostname()
    return rtn


def assert_valid_nid(nid, g):
    assert type(nid) is int and (0 <= nid < g.number_of_nodes())


def assert_0_based_nids(g):
    for i, (n, ndata) in enumerate(sorted(g.nodes(data=True))):
        assert_valid_nid(n, g)
        assert i == n  # 0-based consecutive node ids


def format_str_list(sl):
    assert type(sl) is list
    if not sl:
        return 'None'
    else:
        return ','.join(sl)


class C(object):  # counter
    def __init__(self):
        self.count = 0

    def c(self):  # count and increment itself
        self.count += 1
        return self.count

    def t(self):  # total
        return self.count

    def reset(self):
        self.count = 0


class OurTimer(object):
    def __init__(self):
        self.t = time()
        self.durations_log = OrderedDict()

    def time_and_clear(self, log_str='', only_seconds=False):
        duration = self._get_duration_and_reset()
        if log_str:
            if log_str in self.durations_log:
                raise ValueError('log_str {} already in log {}'.format(
                    log_str, self.durations_log))
            self.durations_log[log_str] = duration
        if only_seconds:
            rtn = duration
        else:
            rtn = format_seconds(duration)
        # print(log_str, '\t\t', rtn)
        return rtn

    def start_timing(self):
        self.t = time()

    def print_durations_log(self):
        print('Timer log', '*' * 50)
        rtn = []
        tot_duration = sum([sec for sec in self.durations_log.values()])
        print('Total duration:', format_seconds(tot_duration))
        lss = np.max([len(s) for s in self.durations_log.keys()])
        for log_str, duration in self.durations_log.items():
            s = '{0}{1} : {2} ({3:.2%})'.format(
                log_str, ' ' * (lss - len(log_str)), format_seconds(duration),
                         duration / tot_duration)
            rtn.append(s)
            print(s)
        print('Timer log', '*' * 50)
        self.durations_log = OrderedDict()  # reset
        return rtn

    def _get_duration_and_reset(self):
        now = time()
        duration = now - self.t
        self.t = now
        return duration

    def get_duration(self):
        now = time()
        duration = now - self.t
        return duration

    def reset(self):
        self.t = time()


def format_seconds(seconds):
    """
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('min', 60),
        ('sec', 1)
    ]

    if seconds <= 1:
        return '{:.3f} msecs'.format(seconds * 1000)

    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            if period_name == 'sec':
                period_value = seconds
                has_s = 's'
            else:
                period_value, seconds = divmod(seconds, period_seconds)
                has_s = 's' if period_value > 1 else ''
            strings.append('{:.3f} {}{}'.format(period_value, period_name, has_s))

    return ', '.join(strings)


def random_w_replacement(input_list, k=1):
    return [random.choice(input_list) for _ in range(k)]


def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.items():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            if n == m:
                assoc[a2idx[a], b2idx[b]] = assoc[b2idx[b], a2idx[a]] = 1.
            else:
                assoc[a2idx[a], b2idx[b]] = 1
    assoc = sp.csr_matrix(assoc)
    return assoc


def prompt(str, options=None):
    while True:
        t = input(str + ' ')
        if options:
            if t in options:
                return t
        else:
            return t


def prompt_get_cpu():
    from os import cpu_count
    while True:
        num_cpu = prompt(
            '{} cpus available. How many do you want?'.format( \
                cpu_count()))
        num_cpu = parse_as_int(num_cpu)
        if num_cpu and num_cpu <= cpu_count():
            return num_cpu


def parse_as_int(s):
    try:
        rtn = int(s)
        return rtn
    except ValueError:
        return None


computer_name = None


def prompt_get_computer_name():
    global computer_name
    if not computer_name:
        computer_name = prompt('What is the computer name?')
    return computer_name


def node_has_type_attrib(g):
    for (n, d) in g.nodes(data=True):
        if 'type' in d:  # TODO: needs to be fixed
            return True
    return False


def print_g(label, g, print_func=print):
    print_func(f'{label} {g.number_of_nodes()} nodes {g.number_of_edges()} edges ({type(g)})')


class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None, bn=False, batched_num=None):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        dims = hidden_channels + [output_dim]
        if batched_num is not None:
            assert type(batched_num) is int and batched_num > 0
            dims = [x * batched_num for x in dims]
        self.layer_channels = [input_dim] + dims
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 1)])))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x, *_, **kwargs):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]


def create_act(act, num_parameters=None):
    if act == 'relu' or act == 'ReLU':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity' or act == 'None':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    if act == 'elu' or act == 'elu+1':
        return nn.ELU()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


def print_stats(li, name, saver=None):
    if saver is None:
        func = print
    else:
        func = saver.log_info
    if len(li) == 0:
        func(f'empty li {name}')
        return
    stats = OrderedDict()
    stats['#'] = len(li)
    stats['Avg'] = np.mean(li)
    stats['Std'] = np.std(li)
    stats['Min'] = np.min(li)
    stats['Max'] = np.max(li)
    stats['Median'] = np.median(li)
    stats['Mode'] = scipy.stats.mode(li)[0][0]
    func(name)
    for k, v in stats.items():
        func(f'\t{k}:\t{v:.4f}')


def plot_dist(data, label, save_dir, saver=None, analyze_dist=True, bins=None):
    if analyze_dist:
        _analyze_dist(saver, label, data)
    fn = f'distribution_{label}.png'
    plt.figure()
    sns.set()
    ax = sns.distplot(data, bins=bins, axlabel=label, kde=False, norm_hist=False)
    plt.xlabel(label)
    ax.figure.savefig(join(save_dir, fn))
    plt.close()


def _analyze_dist(saver, label, data):
    if saver is None:
        func = print
    else:
        func = saver.log_info
    func(f'--- Analyzing distribution of {label} (len={len(data)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    probs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    quantiles = mstats.mquantiles(data, prob=probs)
    func(f'{label} {len(data)}')
    s = '\t'.join(['{:10.4f}'.format(x) for x in probs])
    func(f'\tprob     \t {s}')
    s = '\t'.join(['{:10.4f}'.format(x) for x in quantiles])
    func(f'\tquantiles\t {s}')
    func(f'\t#     \t {len(data)}')
    func(f'\tMin   \t {np.min(data)}')
    func(f'\tMax   \t {np.max(data)}')
    func(f'\tMean  \t {np.mean(data)}')
    func(f'\tStd   \t {np.std(data)}')
    func(f'\tMedian\t {np.median(data)}')
    func(f'\tMode  \t {scipy.stats.mode(data)[0][0]}')


def get_model_info_as_str(FLAGS):
    rtn = []
    d = vars(FLAGS)
    for k in d.keys():
        v = str(d[k])
        if k == 'dataset_list':
            s = '{0:26} : {1}'.format(k, v)
            rtn.append(s)
        else:
            vsplit = v.split(',')
            assert len(vsplit) >= 1
            for i, vs in enumerate(vsplit):
                if i == 0:
                    ks = k
                else:
                    ks = ''
                if i != len(vsplit) - 1:
                    vs = vs + ','
                s = '{0:26} : {1}'.format(ks, vs)
                rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def extract_config_code():
    with open(join(get_src_path(), 'config.py')) as f:
        return f.read()


def plot_scatter_line(data_dict, label, save_dir):
    fn = f'scatter_{label}_iterations.png'
    ss = ['rs-', 'b^-', 'g^-', 'c^-', 'm^-', 'ko-', 'yo-']
    cs = [s[0] for s in ss]
    plt.figure()
    i = 0

    # min_size = min([len(x['incumbent_data']) for x in data_dict.values()])
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li, y_li = [], []

        # min_len = float('inf')
        # for x in data_dict_elt['incumbent_data']:
        #     if x[1] < min_len:
        #         min_len = x[1]

        for x in data_dict_elt['incumbent_data']:
            # if x[1] > FLAGS.recursion_threshold:
            #     break
            x_li.append(x[1])
            y_li.append(x[0])
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()

    plt.figure()
    fn = f'scatter_{label}_time.png'
    i = 0
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li = [x[2] for x in data_dict_elt['incumbent_data']]
        y_li = [x[0] for x in data_dict_elt['incumbent_data']]
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    # plt.close()


POINTS_MARKERS = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
POINTS_COLORS = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige",
                 "brown", "gray", "cyan", "magenta"]


def plot_points(points_dict, label, save_dir):
    i = 0
    for pname, points in points_dict.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                 color=POINTS_COLORS[i % len(POINTS_COLORS)],
                 label=f'{pname}_{label}')
        i += 1
    plt.legend(loc='best')
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()


def multi_plot_dimension(target_list):
    num_figure = len(target_list)
    if num_figure == 1:
        y_dim = 1
        x_dim = 1
    elif num_figure == 2:
        y_dim = 1
        x_dim = 2
    elif num_figure == 3:
        y_dim = 1
        x_dim = 3
    elif num_figure == 4:
        y_dim = 2
        x_dim = 2
    elif num_figure == 5 or num_figure == 6:
        y_dim = 2
        x_dim = 3
    else:
        assert False
    # points_dict = {}
    return num_figure, x_dim, y_dim


def plot_scatter_with_subplot(points_dict_multi_target, label, save_dir, target_list, connected=True):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list)
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    fig.set_figheight(18)
    fig.set_figwidth(24)
    m = {'p': 'o', 't': 'x'}
    for idx, target in enumerate(target_list):
        points_dict[f'p'] = points_dict_multi_target[target]['pred']
        points_dict[f't'] = points_dict_multi_target[target]['true']
        ax = plt.subplot(y_dim, x_dim, idx + 1)
        ax.set_facecolor('xkcd:gray')
        i = 0
        for pname, points_ in points_dict.items():  # dict (true/pred) of dict (name: points)
            for gname, points in points_.items():
                x_li = [str(int(point[0])) for point in sorted(points)]
                y_li = [round(float(point[1]), 2) for point in sorted(points)]
                plt.scatter(np.array(x_li), np.array(y_li), label=f'{gname}-{pname}', color=cs[i % len(cs)],
                            marker=m[pname])
                if connected:
                    plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
                # plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                #         color=POINTS_COLORS[i % len(POINTS_COLORS)],
                #         label=f'{pname}')
                i += 1
        plt.legend(loc='best')
        plt.title(f'{target}')
        plt.grid(True)
        plt.axis('on')
        points_dict = {}

    plt.suptitle(f'{label}')
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()


def plot_scatter_with_subplot_trend(points_dict_multi_target, label, save_dir, target_list, connected=True):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list)
    num_figure, x_dim, y_dim = 1, 1, 1
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    fig.set_figheight(18)
    fig.set_figwidth(24)
    m = {'p': 'o', 't': 'x'}
    for idx, target in enumerate(target_list):
        if 'perf' not in target:
            continue
        # points_dict[f'p'] = points_dict_multi_target[target]['pred']
        points_dict[f't'] = points_dict_multi_target[target]['true']
        ax = plt.subplot(y_dim, x_dim, idx + 1)
        ax.set_facecolor('xkcd:gray')
        i = 0
        for pname, points_ in points_dict.items():  # dict (true/pred) of dict (name: points)
            for gname, points in points_.items():
                if len(points) <= 1:
                    continue
                fig = plt.figure()
                x_li = [str(int(point[0])) for point in sorted(points)]
                y_li = [round(float(point[1]), 2) for point in sorted(points)]
                plt.scatter(np.array(x_li), np.array(y_li), label=f'{gname}-{pname}', color=cs[i % len(cs)],
                            marker=m[pname])
                if connected:
                    plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
                # plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                #         color=POINTS_COLORS[i % len(POINTS_COLORS)],
                #         label=f'{pname}')
                i += 1
                # plt.legend(loc='best')
                plt.title(f'{target}')
                plt.grid(True)
                plt.axis('on')
                # points_dict = {}

                plt.suptitle(f'{label}')
                fn = f'points_{gname}.png'
                plt.savefig(join(save_dir, fn), bbox_inches='tight')
                plt.close()


def plot_points_with_subplot(points_dict_multi_target, label, save_dir, target_list):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list)
    points_dict = {}
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    fig.set_figheight(7.2)
    fig.set_figwidth(10.8)
    for idx, target in enumerate(target_list):
        # points_dict[f'pred_points'] = points_dict_multi_target[target]['pred']
        # points_dict[f'true_points'] = points_dict_multi_target[target]['true']
        pdict = points_dict_multi_target[target]
        plt.subplot(y_dim, x_dim, idx + 1)
        i = 0
        # xs = pdict['true']
        # ys = pdict['pred']
        xs = [(t, t) for t in pdict['true']]
        ys = [(t, p) for t, p in zip(pdict['true'], pdict['pred'])]
        # for pname, points in points_dict_multi_target[target]['pred']:
        #     if pname == 'pred_points':
        #         xs = [point[0] for point in points]
        #         ys = [point[1] for point in points]
        plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                 color=POINTS_COLORS[i % len(POINTS_COLORS)])
        std = pdict.get('pred_std')
        if std is not None:
            plt.errorbar(xs, ys, std, linestyle='None')
            # i += 1

        xpoints = ypoints = plt.xlim()
        plt.plot(xpoints, ypoints, color='green', lw=3, scalex=False, scaley=False)

        # plt.legend(loc='best')
        plt.title(f'{target}')
        # points_dict = {}
    plt.suptitle(f'{label}')
    fn = f'{label}_plot.png'
    fp = join(save_dir, fn)
    create_dir_if_not_exists(dirname(fp))
    plt.savefig(fp, bbox_inches='tight')
    plt.close()


def load_replace_flags(FLAGS):
    from saver import saver
    assert FLAGS.load_model is not None
    p1 = join(dirname(FLAGS.load_model), 'FLAGS.klepto')
    loaded_flags = load(p1)
    p2 = join(dirname(dirname(FLAGS.load_model)), 'FLAGS.klepto')
    if loaded_flags is None:
        loaded_flags = load(p2)
    p3 = join(dirname(dirname(dirname(FLAGS.load_model))), 'FLAGS.klepto')
    if loaded_flags is None:
        loaded_flags = load(p3)
    if loaded_flags is None:
        saver.log_info(f'Tried to load from:\n\t{p1}\n\t{p2}\n\t{p3}')
        raise ValueError('Check load_model: Does this folder/file exsit?')
    loaded_flags = loaded_flags['FLAGS']

    excluded_flags = {'device', 'dataset', 'split_by', 'user', 'hostname', 'ts',
                      'load_model', 'compute_loss_during_testing',
                      'debug_first_train_iters',
                      'total_runtime', 'sample_all_edges', 'priority_correction',
                      'sample_all_edges_thresh', 'plot_final_tree',
                      'time_analysis', 'no_search_tree', 'dataset_list',
                      'num_nodes_max', 'val_every_iter', 'val_debug', 'plot_final_tree',
                      'val_method_list', 'logging',
                      'subtask', 'force_regen', 'train_fashion', 'target', 'num_epoch',
                      'target_preproc', 'num_designs_per_kernel_test', 'num_iters',
                      'adaptation_times', 'test_kernels_in_train', 'progressive_steps', #'test_kernels',
                      'adapt_designs_sample_algo', 'adapt_designs_sample_algo_quality',
                      'task_batch_size', 'test_scenario', 'test_perc',
                      'test_look_for_or_not', 'test_look_for', 'design_db', 'random_seed',
                      'load_db', 'num_tasks_per_kernel_test', 'save_sampled_designs',
                      'itype_mask_perc', 'model_path', #'force_regen_pairwise_data',
                      'epoch_num', 'replace_with_random_weights', 'vis_transformer_att',
                      'encoder_path'
                      # 'learning_algorithm' # TODO: remove this later
                      }
    # TODO: check here to be careful when loading model
    diff_count = 0
    for k in vars(loaded_flags):
        if k in excluded_flags:
            continue
        if not hasattr(FLAGS, k):
            cur_v = None
        else:
            cur_v = getattr(FLAGS, k)
        loaded_v = getattr(loaded_flags, k)
        if cur_v != loaded_v:
            setattr(FLAGS, k, loaded_v)
            saver.log_info('\t{}={}\n\t\tto {}={}'.format(k, cur_v, k, loaded_v))
            diff_count += 1
            continue
    saver.log_info(f'Done loading FLAGS diff_count {diff_count}')
    # exit(-1)


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


class TopKModelMaintainer(object):
    def __init__(self, K):
        assert K >= 1
        self.K = K
        self.li = []

    def add_model(self, model, iteration, val_loss, save_dir):

        self.li.append([val_loss, iteration, None])

        self.li.sort()

        i = self._need_save(iteration)
        if i is not None:
            from saver import saver

            saver.log_info(f'TopKModelMaintainer: In top {self.K}: '
                           f'Saved model at iter {iteration} with val loss {val_loss:.4f}')

            save_file = Path(f'{save_dir}/model_{iteration}_{val_loss:.4f}.pth')
            torch.save(model.state_dict(), save_file)
            self.li[i][2] = save_file

            if len(self.li) > self.K:
                to_remove = self.li[self.K]
                remove(to_remove[2])
                saver.log_info(f'TopKModelMaintainer: '
                               f'Removed {to_remove[2]} from save folder')

    def _need_save(self, iteration):
        if len(self.li) > self.K:
            for i in range(self.K):
                if self.li[i][1] == iteration:
                    # Current model val_loss is in top K.
                    return i
            return None
        else:
            for i in range(len(self.li)):
                if self.li[i][1] == iteration:
                    return i
            assert False


def get_gname(data):
    return data.gname.split('_')[0]


def check_prepend_root_folder(path):
    if path and path[0] != '/':
        return join(get_src_path(), 'logs', path)
    else:
        return path


def get_best_gpu(option, total_num_gpus=8, print_func=print):
    import torch
    if not torch.cuda.is_available():
        return -1
    import nvidia_smi

    gpus = []
    for gpu in range(total_num_gpus):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        # print("Total memory:", info.total)
        # print("Free memory:", info.free)
        # print("Used memory:", info.used)

        nvidia_smi.nvmlShutdown()
        gpus.append((info.free, gpu))


    assert len(gpus) == total_num_gpus
    gpus.sort(reverse=True)

    def format_file_size(size, decimals=2, binary_system=True):
        if binary_system:
            units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB']
            largest_unit = 'YiB'
            step = 1024
        else:
            units = ['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']
            largest_unit = 'YB'
            step = 1000

        for unit in units:
            if size < step:
                return ('%.' + str(decimals) + 'f %s') % (size, unit)
            size /= step

        return ('%.' + str(decimals) + 'f %s') % (size, largest_unit)

    for free_mem, gpu in gpus:
        print_func(f'gpu {gpu} free {format_file_size(free_mem)}')

    if option == 'user_input':
        done = False
        while not done:
            choose = input("Enter Integer Number!\n")
            print_func(f"You entered {choose}")
            try:
                choose = int(choose)
                if 0 <= choose <= 7:
                    done = True
                else:
                    print_func(f"Not a valid chosen gpu id")
            except ValueError:
                print_func(f"Not an integer")

    elif option == 'auto':
        choose = gpus[0][1]
        print_func(f'Choose gpu {choose}')
    else:
        raise NotImplementedError()
    return choose


def create_pred_dict(target_list, extra_entries=None):
    rtn = OrderedDict()
    for target_name in target_list:
        rtn[target_name] = {'true': [], 'pred': []}
        if extra_entries is not None:
            for e in extra_entries:
                rtn[target_name][e] = []
    return rtn


def create_edge_index(g): # TODO: check directions
    # g = nx.convert_node_labels_to_integers(g, ordering='sorted')
    edge_index = torch.LongTensor(list(g.edges)).t().contiguous() # critical tn ensure g.edges ordering agrees with edge_attr generation code
    return edge_index



NON_OPT_PRAGMAS = ['LOOP_TRIPCOUNT', 'INTERFACE', 'INTERFACE', 'KERNEL']
WITH_VAR_PRAGMAS = ['DEPENDENCE', 'RESOURCE', 'STREAM', 'ARRAY_PARTITION']
# TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']

def check_any_in_str(li, s):
    for li_item in li:
        if li_item in s:
            return True
    return False

def coo_to_sparse(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    rtn = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return rtn



def estimate_model_size(model, label, saver):
    def _to_MB(s):
        return s / 1024 ** 2

    param_size = 0
    num_params = 0
    sizes = []
    for name, param in model.named_parameters():
        param_size_this = param.nelement() * param.element_size()
        sizes.append((param_size_this, name))
        param_size += param_size_this
        num_params += 1
    sizes.sort(reverse=True)
    saver.log_info('Printing top 10 large model parameters:')
    for size, name in sizes[0:10]:
        saver.log_info(f'\t{name} size: {_to_MB(size):.3f}MB')
    buffer_size = 0
    num_buffers = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        num_buffers += 1

    size_all_mb = _to_MB(param_size + buffer_size)
    saver.log_info(f'The {label} has {num_params} parameters, {num_buffers} buffers of (estimated) size: {size_all_mb:.3f}MB')





def create_loss_dict_get_target_list(FLAGS):
    loss_dict = {}
    # i = 0
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task == 'regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]
    # target_list = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
    if hasattr(FLAGS, 'node_token_alignment_loss') and FLAGS.node_token_alignment_loss:
        loss_dict['loss_pc_links'] = 0.0
    if hasattr(FLAGS, 'gs_alignment_loss') and FLAGS.gs_alignment_loss:
        loss_dict['loss_gs_alignment'] = 0.0
    for t in target_list:
        loss_dict[t] = 0.0
    return loss_dict, target_list


def update_loss_dict(loss_dict, loss_dict_, target_list, FLAGS):
    for t in target_list:
        loss_dict[t] += loss_dict_[t].item()
    if hasattr(FLAGS, 'node_token_alignment_loss') and FLAGS.node_token_alignment_loss:
        loss_dict['loss_pc_links'] += loss_dict_['loss_pc_links'].item()
    if hasattr(FLAGS, 'gs_alignment_loss') and FLAGS.gs_alignment_loss:
        loss_dict['loss_gs_alignment'] += loss_dict_['loss_gs_alignment'].item()
    if 'guide_loss' in loss_dict_:
        if 'guide_loss' not in loss_dict:
            loss_dict['guide_loss'] = 0
        loss_dict['guide_loss'] += loss_dict_['guide_loss'].item()
    return loss_dict
