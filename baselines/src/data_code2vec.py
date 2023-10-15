from config import FLAGS
from saver import saver
from utils import get_root_path, load_pickle
from torch_geometric.data import Data
import torch
from collections import defaultdict
from os.path import join
import numpy as np


class Code2VecData(object):
    pass


did_to_emb_dict = {}
did_to_point_dict = {}
kernel_design_lookup_dict = defaultdict(dict)

miss_did_cnt = 0


def _load_code2vec_data():
    global did_to_emb_dict
    global did_to_point_dict
    global kernel_design_lookup_dict
    global miss_did_cnt
    if FLAGS.v2_db:
        ver = 'v20'
    else:
        ver = 'v18'

    if not did_to_emb_dict or not did_to_point_dict:
        p = join(get_root_path(), 'file', 'code2vec', FLAGS.code2vec_data_version, f'{ver}-emb', 'emb.pkl')
        did_to_emb_dict = load_pickle(p)
        if not did_to_emb_dict:
            raise FileNotFoundError(f'Cannot load did_to_emb_dict from {p}; double check the path')
        p = join(get_root_path(), 'file', 'code2vec', FLAGS.code2vec_data_version, f'{ver}-emb', 'fname_to_config.pkl')
        did_to_point_dict = load_pickle(p)
        if not did_to_point_dict:
            raise FileNotFoundError(f'Cannot load did_to_point_dict from {p}; double check the path')
        if len(did_to_emb_dict) != len(did_to_point_dict):
            saver.log_info(
                f'len(did_to_emb_dict)={len(did_to_emb_dict)} != len(did_to_point_dict)={len(did_to_point_dict)}')
        for did, emb in did_to_emb_dict.items():
            spt = did.split('_')
            gname, _ = spt[0], spt[1]
            did = did.split('.c')[0]
            point = did_to_point_dict.get(did)
            if point is None:
                miss_did_cnt += 1
                saver.log_info(f'miss_did_cnt={miss_did_cnt} {did}!')
            else:
                kernel_design_lookup_dict[gname][str(point)] = emb

        saver.log_info(
            f'Loaded {len(kernel_design_lookup_dict)} kernels\'s {len(did_to_emb_dict)} designs from disk: {kernel_design_lookup_dict.keys()}')


def init_preprocessors_code2vec():
    return {'encoders': {}}


def read_code2vec_data(gexf_file):
    return Code2VecData()


def encode_feat_dict_code2vec(mmo, preprocessors, point=None):
    return {}, {}


def fit_preprocessors_code2vec(preprocessors):
    pass


not_found_cnt = 0


def encode_X_torch_code2vec(g, d_node, d_edge, preprocessors, gname, vname, return_Data=True):
    global not_found_cnt
    _load_code2vec_data()

    point_this_design = str(d_node['point'])
    emb = kernel_design_lookup_dict[gname].get(point_this_design)
    if emb is None:
        not_found_cnt += 1
        saver.log_info(f'not_found_cnt={not_found_cnt} {gname}\'s design {point_this_design} not found in code2vec!')
        emb = np.ones((384,))
    assert emb.shape == (384,)
    emb = torch.tensor(emb).view(1, -1)

    if return_Data:
        d_node.pop('point')

        if FLAGS.task == 'regression':
            return Data(
                gname=gname,
                key=vname,

                x=emb,

                perf=d_node['perf'],
                actual_perf=d_node['actual_perf'],
                kernel_speedup=d_node['kernel_speedup'],
                quality=d_node['quality'],
                util_BRAM=d_node['util-BRAM'],
                util_DSP=d_node['util-DSP'],
                util_LUT=d_node['util-LUT'],
                util_FF=d_node['util-FF'],
                total_BRAM=d_node['total-BRAM'],
                total_DSP=d_node['total-DSP'],
                total_LUT=d_node['total-LUT'],
                total_FF=d_node['total-FF'],
            )
        elif FLAGS.task == 'class':
            return Data(
                gname=gname,
                key=vname,

                x=emb,

                perf=d_node['perf'],
            )
        else:
            raise NotImplementedError()




    else:
        assert False
