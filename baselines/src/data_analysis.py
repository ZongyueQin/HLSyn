from model import Net, get_y_with_target
from config import FLAGS
import torch
from torch_geometric.data import DataLoader
from saver import saver

from os.path import join
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

import learn2learn as l2l
import numpy as np
from collections import defaultdict

from tqdm import tqdm

from torch_scatter import scatter_add


def analyze_data(dataset):
    setattr(FLAGS, 'target', ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF'])
    # model = Net(dataset.num_features).to(FLAGS.device)
    # print(model)
    # print(model.state_dict().keys())
    # assert FLAGS.load_model != 'None'
    # if not FLAGS.only_pragma_nodes:
    #     saver.log_info(f'Loading model: {FLAGS.load_model}')
    #     ld = torch.load(FLAGS.load_model, map_location=FLAGS.device)
    #
    #     if FLAGS.MAML and FLAGS.learning_algorithm == 'MAML':
    #         model = l2l.algorithms.MAML(model, lr=FLAGS.fast_lr, first_order=False)
    #
    #     model.load_state_dict(ld)
    #     saver.log_info('Model loaded')
    # else:
    #     saver.log_info('Skipping loading model since only pragma nodes')

    setattr(FLAGS, 'target',
            ['perf', 'actual_perf', 'util_DSP', 'util_BRAM', 'util_LUT', 'util_FF'])
    designs_by_kernel = defaultdict(list)
    for i, data in enumerate(tqdm(dataset)):
        for target in FLAGS.target:
            setattr(data, target, getattr(data, target).item())
        designs_by_kernel[data.gname].append(data)

    for kernel in ['spmv-crs', 'spmv-ellpack']:
        designs = designs_by_kernel[kernel]
        saver.log_info(f'Found {len(designs)} designs for kernel {kernel}')
        designs.sort(key=lambda x: x.perf, reverse=True)
        for j, d in enumerate(designs):
            saver.log_info(f'\t{j}: {d}')
            if j == 3:
                break
