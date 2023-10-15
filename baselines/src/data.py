from database import create_database
from data_code2vec import init_preprocessors_code2vec, read_code2vec_data, encode_feat_dict_code2vec, \
    fit_preprocessors_code2vec, encode_X_torch_code2vec
from data_multi_modality import init_preprocessors_multi_modality, read_multi_modality_data, \
    encode_feat_dict_multi_modality, \
    fit_preprocessors_multi_modality, encode_X_torch_multi_modality
from data_programl import init_preprocessors_programl, read_programl_graph, encode_feat_dict_programl, \
    fit_preprocessors_programl, encode_X_torch_programl
from data_src_code import init_preprocessors_src_code, read_source_code, encode_feat_dict_src_code, \
    fit_preprocessors_src_code, encode_X_torch_src_code
from graph_transfotmer import collate_batch_graph_transformer

from config import FLAGS
from saver import saver
from utils import get_root_path, MLP, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, save_pickle, load_pickle, load, create_edge_index, sorted_nicely, \
    coo_to_sparse, save
from result import Result, persist

from os.path import join, basename, dirname
from glob import glob, iglob

from math import ceil

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

import redis, pickle, random
import numpy as np
from collections import Counter, defaultdict, OrderedDict

from scipy.sparse import hstack

from tqdm import tqdm

import os.path as osp

import torch
from torch_geometric.data import Dataset

from shutil import rmtree
from torch.utils.data import random_split
import math

KEY_PRAGMA = '__PIPE__L0'
FURTHER_SPLIT = ''
if FLAGS.subtask == 'train' and FLAGS.MAML:
    if FLAGS.MAML_split_kernel:
        FURTHER_SPLIT = f'split_{KEY_PRAGMA}'
    else:
        FURTHER_SPLIT = f'nosplit'
else:
    FURTHER_SPLIT = f'nosplit'


def _get_save_dir(FLAGS):
    if hasattr(FLAGS, 'force_data_dir'):
        saver.log_info(f'FLAGS has force_data_dir; just use that')
        return FLAGS.force_data_dir

    if FLAGS.encoder_path is not None and FLAGS.encoder_path != 'None':
        encoder_str = FLAGS.load_encoders_label
    else:
        encoder_str = 'regular_encoder'
        # encoder_str = 'None'

    need_shrinking = False
    if FLAGS.v2_db:
        if FLAGS.only_common_db:
            if FLAGS.test_extra:
                SAVE_DIR = join(get_save_path(), FLAGS.dataset,
                                f'with-updated-task-transfer-new-db-new-speedup-common-only-extra-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
            else:
                SAVE_DIR = join(get_save_path(), FLAGS.dataset,
                                f'with-updated-task-transfer-new-db-new-speedup-common-only-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
        else:
            if FLAGS.encoder_path is None:
                SAVE_DIR = join(get_save_path(), FLAGS.dataset,
                                f'with-updated-new-db-new-speedup-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
            else:
                if not FLAGS.only_new_points:
                    SAVE_DIR = join(get_save_path(), FLAGS.dataset,
                                    f'v2-transfer-round{FLAGS.round_num}-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
                else:
                    SAVE_DIR = join(get_save_path(), FLAGS.dataset,
                                    f'with-updated-task-transfer-dse1-only-new-db-new-speedup-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}',
                                    'dse3')
    else:
        if FLAGS.new_speedup:
            # SAVE_DIR = join(get_save_path(), FLAGS.dataset,
            #                 f'round{FLAGS.round_num}-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
            SAVE_DIR = join(get_save_path(), FLAGS.dataset,
                            f'r{FLAGS.round_num}-ifdb-{FLAGS.graph_type}-{FLAGS.task}_ep-{FLAGS.encode_edge_position}_nowi_{FLAGS.invalid}-n_{FLAGS.norm_method}_np_{FLAGS.no_pragma}_{FLAGS.tag}')
        else:
            # SAVE_DIR = join(get_save_path(), FLAGS.dataset,
            #                 f'with-updated-task-transfer-old-db-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
            SAVE_DIR = join(get_save_path(), FLAGS.dataset,
                            f'wutfod-{FLAGS.task}_wi_{FLAGS.invalid}-n_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_{FLAGS.tag}')

    if FLAGS.model == 'code2vec':
        rtn = f'{SAVE_DIR}_code2vec-{FLAGS.code2vec_data_version}'
        return rtn
    assert FLAGS.model == 'our'

    ptrans_s = ''
    if FLAGS.ptrans:
        ptrans_s = f'_{FLAGS.pragma_scope}'
    smodeling_s = ''
    if FLAGS.sequence_modeling:
        # smodeling_s = f'_seq_{FLAGS.code_encoder}'
        smodeling_s = f'_s_{FLAGS.data_repr}_{FLAGS.code_encoder}'
        if hasattr(FLAGS, 'max_code_tokens_len') and FLAGS.max_code_tokens_len is not None:
            smodeling_s = f'{smodeling_s}_{FLAGS.max_code_tokens_len}'
            if hasattr(FLAGS, 'bi_directional_AST') and FLAGS.bi_directional_AST:
                smodeling_s = f'{smodeling_s}_bi'
        if hasattr(FLAGS, 'token_att_masking') and FLAGS.token_att_masking:
            smodeling_s = f'{smodeling_s}_tm'
        if hasattr(FLAGS, 'token_att_masking') and FLAGS.preserve_keywords:
            smodeling_s = f'{smodeling_s}_pk'
            if hasattr(FLAGS, 'pk_version') and FLAGS.pk_version >= 2:
                smodeling_s = f'{smodeling_s}_v{FLAGS.pk_version}'
        if hasattr(FLAGS, 'add_edges_fc_graph') and FLAGS.add_edges_fc_graph:
            smodeling_s = f'{smodeling_s}_fc'
        if hasattr(FLAGS, 'chunk_offset') and FLAGS.chunk_offset != 0:
            smodeling_s = f'{smodeling_s}_co{FLAGS.chunk_offset}'

    graph_type_s = ''
    if FLAGS.graph_type != '':
        if FLAGS.graph_type not in SAVE_DIR:
            graph_type_s = f'_{FLAGS.graph_type}'

    ignore_kernels_s = ''
    if hasattr(FLAGS, 'ignore_kernels') and len(FLAGS.ignore_kernels) > 0:
        s = '_'.join([str(x) for x in FLAGS.ignore_kernels])
        ignore_kernels_s = f'_ig={s}'

    mms = ''
    if FLAGS.multi_modality:
        mms = f'_{FLAGS.what_modalities}'
        if FLAGS.combine_fashion == 'share_final_MLPs':
            if FLAGS.feed_p_to_tf:
                mms = f'{mms}_feed'
            if FLAGS.pc_links:
                mms = f'{mms}_pclcc'  # 05/05/2023: found the token_id_global_in_chunks bug (need to handle eos token) Although this only affects node_token_interaction, still change the dataset name to regenerate.
                # mms = f'{mms}_pclc'  # 04/30/2023: c means "correct" with pragma nodes
                # mms = f'{mms}_pcl'
                need_shrinking = True
                if FLAGS.pc_links_aug is not None:
                    mms = f'{mms}_{FLAGS.pc_links_aug}'
                if FLAGS.node_token_interaction:
                    mms = f'{mms}_ntic'  # 05/05/2023: found the token_id_global_in_chunks bug (need to handle eos token)
                    # mms = f'{mms}_nti'
                if FLAGS.pc_links_holdout_ratio > 0:
                    mms = f'{mms}_{FLAGS.pc_links_holdout_ratio}'
                if hasattr(FLAGS, 'interleave_GNN_transformer') and FLAGS.interleave_GNN_transformer:
                    mms = f'{mms}_igt'
        elif FLAGS.combine_fashion == 'share_GNNs_MLPs':
            need_shrinking = True
            mms = f'{mms}_{FLAGS.combine_fashion}'
            if hasattr(FLAGS, 'add_pragma_links') and FLAGS.add_pragma_links:
                mms = f'{mms}_apl'
        # elif FLAGS.combine_fashion == 'p_to_tf':
        #     need_shrinking = True
        #     mms = f'{mms}_{FLAGS.combine_fashion}'
        #     mms = f'{mms}_{FLAGS.feed_what}'
        else:
            raise NotImplementedError()

    combine_ast_programl = False  # deprecated flag
    rtn = f'{SAVE_DIR}_{FLAGS.gtype}_{FLAGS.only_pragma_nodes}_{combine_ast_programl}_' \
          f'{FLAGS.encode_full_text}_{FLAGS.fulltext_dim}_{FURTHER_SPLIT}_{encoder_str}_{FLAGS.ptrans}{ptrans_s}{smodeling_s}{graph_type_s}{ignore_kernels_s}{mms}'

    if need_shrinking:
        len_before = len(rtn)
        rtn = rtn.replace('with-updated-new-db-new-speedup', 'w')
        rtn = rtn.replace('with-updated-task-transfer-new-db-new-speedup-common-only-extra', 'wt')
        rtn = rtn.replace('with-updated-task-transfer-dse1-only-new-db-new-speedup', 'wtd')
        rtn = rtn.replace('with-updated-task-transfer-old-db', 'wto')
        rtn = rtn.replace('normalization', 'n')
        saver.log_info(f'Shrink SAVE_DIR which is too long from {len_before} to {len(rtn)}')

    if len(rtn) >= 199:
        len_before = len(rtn)
        rtn = rtn.replace('v2-transfer-round2-icmp-feb-db--regression_edge-position', 'v2trifdrep')
        rtn = rtn.replace('v2-transfer-round2-icmp-feb-db--class_edge-position', 'v2trifdcep')
        rtn = rtn.replace('v2-transfer-round3-icmp-feb-db--regression_edge-position', 'v2tr3ifdrep')
        rtn = rtn.replace('v2-transfer-round3-icmp-feb-db--class_edge-position', 'v2tr3ifdcep')
        saver.log_info(f'Further Shrink SAVE_DIR which is too long from {len_before} to {len(rtn)}')
        if len(rtn) >= 302:
            len_before = len(rtn)
            rtn = rtn.replace(
                'v2trifdrep-False_norm_with-invalid_False-n_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_programl_False_False_None_None_nosplit',
                'v2trifdrep-s')
            rtn = rtn.replace(
                'v2trifdcep-False_norm_with-invalid_False-n_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_programl_False_False_None_None_nosplit',
                'v2trifdcep-s')
            rtn = rtn.replace(
                'v2tr3ifdrep-False_norm_with-invalid_False-n_speedup-log2_no_pragma_False_tag_wmp-d_programl_False_False_None_None_nosplit',
                'v2trifdrep-sd')
            rtn = rtn.replace(
                'v2tr3ifdrep-False_norm_with-invalid_False-n_speedup-log2_no_pragma_False_tag_wmp-d_programl_False_False_None_None_nosplit',
                'v2trifdcep-sd')
            saver.log_info(f'Further Shrink SAVE_DIR which is too long from {len_before} to {len(rtn)}')

    saver.log_info(f'len(rtn)={len(rtn)}')
    saver.log_info(f'rtn={rtn}')
    return rtn


SAVE_DIR = _get_save_dir(FLAGS)


def update_save_dir(FLAGS):
    global SAVE_DIR, ENCODER_PATH  # tricky; if loaded flags are different; need to call this function to load the correct dataset
    SAVE_DIR = _get_save_dir(FLAGS)
    ENCODER_PATH = join(SAVE_DIR, 'encoders')
    saver.log_info(f'SAVE_DIR and ENCODER_PATH updated due to FLAGS being potentially loaded')
    saver.log_info(f'SAVE_DIR={SAVE_DIR}')


def update_save_dir_with_additional_str(additional_str):
    global SAVE_DIR, ENCODER_PATH
    SAVE_DIR = f'{SAVE_DIR}_{additional_str}'
    ENCODER_PATH = join(SAVE_DIR, 'encoders')
    saver.log_info(f'SAVE_DIR updated with additional str={SAVE_DIR}\nENCODER_PATH={ENCODER_PATH}')


ENCODER_PATH = join(SAVE_DIR, 'encoders')
# PROCESSED_DIR = join(SAVE_DIR, 'processed')
create_dir_if_not_exists(SAVE_DIR)

# DATASET = 'cnn1'
DATASET = 'machsuite-poly'
dse_database_name = FLAGS.dse_database_name
if DATASET == 'cnn1':
    KERNEL = 'cnn'
    db_path = f'../{dse_database_name}/databases/cnn_case1/'
elif DATASET == 'machsuite':
    KERNEL = FLAGS.tag
    db_path = f'../{dse_database_name}/machsuite/databases/**/*'
elif DATASET == 'machsuite-poly':
    KERNEL = FLAGS.tag
    db_path = []
    for benchmark in FLAGS.benchmarks:
        db_path.append(f'../{dse_database_name}/{benchmark}/databases/**/*')

if FLAGS.dataset == 'vitis-cnn':
    GEXF_FOLDER = join(get_root_path(), dse_database_name , 'dotGenerator_all_kernels')
elif FLAGS.dataset == 'machsuite':
    GEXF_FOLDER = join(get_root_path(), dse_database_name, 'machsuite', 'dot-files')
elif FLAGS.dataset == 'programl':
    GEXF_FOLDER = join(get_root_path(), dse_database_name, 'programl', '**', 'processed', '**')
    if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
        GEXF_FOLDER = join(get_root_path(), dse_database_name, 'gexf_with_line_col', '**')
elif FLAGS.dataset == 'programl-machsuite':
    GEXF_FOLDER = join(get_root_path(), dse_database_name, 'programl', 'machsuite', 'processed')
elif FLAGS.dataset == 'simple-programl':
    GEXF_FOLDER = join(get_root_path(), dse_database_name, 'simple-program', 'programl', 'processed', '**')
else:
    raise NotImplementedError()

if FLAGS.check_release_db:
    GEXF_FOLDER = join(get_root_path(), dse_database_name, '**')

import config

TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL

if FLAGS.all_kernels:
    if FLAGS.graph_type == '':
        GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if
                             f.endswith('.gexf') and 'extended' not in f and 'processed' in f])
        if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
            GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if
                                 f.endswith('.gexf') and 'extended' not in f])
    else:
        if 'hierarchy' not in FLAGS.graph_type:
            pruner = 'hierarchy'
        else:
            pruner = 'initial'
        GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if
                             f.endswith('.gexf') and FLAGS.graph_type in f and 'processed' in f and pruner not in f])
else:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith(
        '.gexf') and f'{FLAGS.target_kernel}_' in f and 'extended' not in f and 'processed' in f])

if FLAGS.force_regen:
    assert len(GEXF_FILES) >= 1

# debugging code

# n = []
# for gexf_file in GEXF_FILES:
#     # if 'bicg' in gexf_file and 'large' not in gexf_file  and 'medium' not in gexf_file:
#     #     n.append(gexf_file)
#     if 'stencil-3d' in gexf_file:
#         n.append(gexf_file)
# GEXF_FILES = n

torch_generator = None

if FLAGS.fix_randomness:
    saver.log_info('Critical! data.py: Fix random seed for torch data loader generator')
    torch_generator = torch.Generator()
    torch_generator.manual_seed(0)


def finte_diff_as_quality(new_result: Result, ref_result: Result) -> float:
    """Compute the quality of the point by finite difference method.

    Args:
        new_result: The new result to be qualified.
        ref_result: The reference result.

    Returns:
        The quality value (negative finite differnece). Larger the better.
    """

    def quantify_util(result: Result) -> float:
        """Quantify the resource utilization to a float number.

        util' = 5 * ceil(util / 5) for each util,
        area = sum(2^1(1/(1-util))) for each util'

        Args:
            result: The evaluation result.

        Returns:
            The quantified area value with the range (2*N) to infinite,
            where N is # of resources.
        """

        # Reduce the sensitivity to (100 / 5) = 20 intervals
        utils = [
            5 * ceil(u * 100 / 5) / 100 + FLAGS.epsilon for k, u in result.res_util.items()
            if k.startswith('util')
        ]

        # Compute the area
        return sum([2.0 ** (1.0 / (1.0 - u)) for u in utils])

    ref_util = quantify_util(ref_result)
    new_util = quantify_util(new_result)

    # if (new_result.perf / ref_result.perf) > 1.05:
    #     # Performance is too worse to be considered
    #     return -float('inf')

    if new_util == ref_util:
        if new_result.perf < ref_result.perf:
            # Free lunch
            # return float('inf')
            return FLAGS.max_number
        # Same util but slightly worse performance, neutral
        return 0

    return -(new_result.perf - ref_result.perf) / (new_util - ref_util)


class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_files=None, need_encoders=True, need_attribs=True):
        # self.processed_dir = PROCESSED_DIR
        # print('SAVE_DIR', SAVE_DIR)
        super(MyOwnDataset, self).__init__(SAVE_DIR, transform, pre_transform)
        self.scalers = None
        if need_encoders:
            self.encoders = load_encoders()
        if need_attribs:
            self.attribs = load_pickle(self._get_attribs_sp(), print_msg=False)
            if self.attribs is None:
                self.attribs = {}
        if data_files is not None:
            self.data_files = data_files

    @property
    def raw_file_names(self):
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        # return ['data_1.pt', 'data_2.pt', ...]
        if hasattr(self, 'data_files'):
            return self.data_files
        else:
            rtn = sorted_nicely(glob(join(SAVE_DIR, '*.pt')))
            return rtn

    def add_target_scalers(self, scalers, to_save):
        assert type(scalers) is dict
        self.scalers = scalers
        sp = self._get_scalers_sp()
        if to_save:
            save_pickle(scalers, sp)

    def add_attribute(self, attr_name, attr):
        if attr_name in self.attribs:
            raise RuntimeError(f'Attribute {attr_name} already exist')
        self.attribs[attr_name] = attr
        save_pickle(self.attribs, self._get_attribs_sp())

    def get_attribute(self, attr_name):
        rtn = self.attribs.get(attr_name)
        if rtn is None:
            raise RuntimeError(f'Attribute {attr_name} does not exist')
        return rtn

    def get_target_scalers(self):
        if self.scalers is None:
            sp = self._get_scalers_sp()
            self.scalers = load_pickle(sp)
            if type(self.scalers) is not dict:
                raise ValueError(f'Cannot load scalers from {sp}')
        return self.scalers

    def _get_scalers_sp(self):
        return join(SAVE_DIR, 'scalers.pkl')

    def _get_attribs_sp(self):
        return join(SAVE_DIR, 'attribs.pkl')

    def download(self):
        pass

    # Download to `self.raw_dir`.

    def process(self):
        # i = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     data = Data(...)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
        #     i += 1
        pass

    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()

    def get(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = osp.join(SAVE_DIR, 'data_{}.pt'.format(idx))
        data = torch.load(fn)
        return data


def split_dataset(dataset, torch_generator):
    file_li = dataset.processed_file_names
    transductive_test_file_li = []
    if FLAGS.tvt_split_by == 'designs_transductive':
        saver.log_info(f'tvt_split_by {FLAGS.tvt_split_by} so just mix all designs together and split')
        val_ratio = FLAGS.val_ratio
        test_ratio = FLAGS.val_ratio
        num_graphs = len(dataset)

        if FLAGS.sample_finetune:
            r1 = int(num_graphs * 1.0)
            r2 = int(num_graphs * 0)
        else:
            r1 = int(num_graphs * (1.0 - (val_ratio + test_ratio)))
            r2 = int(num_graphs * val_ratio)

        # file_li = ['xxxx'] * len(file_li)
        lengths = [r1, r2, len(dataset) - r1 - r2]
        if FLAGS.shuffle:
            li = random_split(file_li, lengths,
                              generator=torch.Generator().manual_seed(100))
        else:
            li = split_li(file_li, lengths)
    elif FLAGS.tvt_split_by == 'kernels_inductive':
        saver.log_info(f'tvt_split_by {FLAGS.tvt_split_by} so going through all data points and gather stats first')
        # train_file_li = []
        val_test_kernel_li_map = defaultdict(list)
        train_kernel_li_map = defaultdict(list)
        kernel_names = set()
        for file in tqdm(file_li):
            data = torch.load(file)
            if data.gname in FLAGS.test_kernels:
                val_test_kernel_li_map[data.gname].append(file)
            else:
                train_kernel_li_map[data.gname].append(file)
                # train_file_li.append(file)
            kernel_names.add(data.gname)
        saver.log_info(f'-' * 20)
        # saver.log_info(f'Found {len(train_file_li)} files for training')
        saver.log_info(f'We have {len(kernel_names)} kernels in total: {sorted(kernel_names)}')

        train_file_li = []
        val_file_li = []
        test_file_li = []
        transductive_test_file_li = []

        for gname, gname_train_file_li in sorted(train_kernel_li_map.items()):
            r1 = int(len(gname_train_file_li) * FLAGS.val_ratio_in_train_kernels)
            r2 = int(len(gname_train_file_li) * FLAGS.test_ratio_in_train_kernels)

            if FLAGS.shuffle:
                random.Random(123).shuffle(gname_train_file_li)

            lengths = [r1, r2, len(gname_train_file_li) - r1 - r2]
            li = split_li(gname_train_file_li, lengths)
            saver.log_info(
                f'\tval/train: {gname} has {len(gname_train_file_li)} designs in total; split into val {len(li[0])}, '
                f'transductive test {len(li[1])}, and train {len(li[2])}')
            val_file_li += li[0]
            transductive_test_file_li += li[1]
            train_file_li += li[2]

        for gname, val_test_file_li in sorted(val_test_kernel_li_map.items()):
            r1 = int(len(val_test_file_li) * FLAGS.val_ratio_in_test_kernels)
            if FLAGS.shuffle:
                random.Random(123).shuffle(val_test_file_li)
            lengths = [r1, len(val_test_file_li) - r1]
            li = split_li(val_test_file_li, lengths)
            saver.log_info(
                f'\tval/test: \t{gname} has {len(val_test_file_li)} designs in total; split into val {len(li[0])} and test {len(li[1])}')
            val_file_li += li[0]
            test_file_li += li[1]

        saver.log_info(f'-' * 20)

        li = (train_file_li, val_file_li, test_file_li)
    else:
        assert False

    saver.log_info(f'{len(file_li)} graphs in total:'
                   f' {len(li[0])} train ({len(li[0]) / len(file_li):.2%});'
                   f' {len(li[1])} val ({len(li[1]) / len(file_li):.2%});'
                   f' {len(li[2])} test ({len(li[2]) / len(file_li):.2%})')
    if len(transductive_test_file_li) > 0:
        saver.log_info(
            f' {len(transductive_test_file_li)} transductive test ({len(transductive_test_file_li) / len(file_li):.2%})')

    train_loader = create_dataloader(li[0], shuffle=FLAGS.shuffle)
    val_loader = create_dataloader(li[1], shuffle=False)
    test_loader = create_dataloader(li[2], shuffle=False)

    transductive_test_loader = None
    if len(transductive_test_file_li) > 0:
        transductive_test_loader = create_dataloader(transductive_test_file_li, shuffle=False)
    return train_loader, val_loader, test_loader, transductive_test_loader


def create_dataloader(data_file_li, shuffle):
    dataset = MyOwnDataset(data_files=data_file_li)
    if FLAGS.multi_modality:
        if FLAGS.combine_fashion == 'share_final_MLPs':
            follow_batch = ['x_programl', 'x_src_code']
            if hasattr(FLAGS, 'collaboration_btw_modalities') and FLAGS.collaboration_btw_modalities == 'edge_msgs':
                follow_batch += ['pos_node_ids_train', 'pos_node_ids']
        elif FLAGS.combine_fashion == 'share_GNNs_MLPs':
            follow_batch = ['x_programl', 'x_src_code', 'x_dummy']
        else:
            raise NotImplementedError()
    else:
        follow_batch = None
    if FLAGS.graph_transformer_option is None:
        rtn = PyGDataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=shuffle,
                            num_workers=FLAGS.data_loader_num_workers,
                            pin_memory=True,
                            generator=torch_generator, follow_batch=follow_batch)  # TODO
    else:
        rtn = TorchDataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=shuffle,
                              num_workers=FLAGS.data_loader_num_workers,
                              generator=torch_generator,
                              collate_fn=collate_batch_graph_transformer)  # TODO
    return rtn


def split_li(li, lengths):
    # Taken from python 3.5 docs
    def _accumulate(iterable, fn=lambda x, y: x + y):
        'Return running totals'
        # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return
        yield total
        for element in it:
            total = fn(total, element)
            yield total

    if sum(lengths) != len(li):
        raise ValueError(
            f"Sum of input lengths does not equal the length of the input dataset! lengths={lengths}; len(li)={len(li)}")

    return [li[offset - length: offset] for offset, length in zip(_accumulate(lengths), lengths)]


def get_kernel_samples(dataset):
    samples = defaultdict(list)
    for data in dataset:
        if FLAGS.target_kernel in data.gname:
            samples[FLAGS.target_kernel].append(data)

    return samples[FLAGS.target_kernel]


def encode_from_db(kernel_name, db_file_path):
    gexf_file = None
    for file in GEXF_FILES:
        if kernel_name in file:
            assert gexf_file is None
            gexf_file = file
    assert gexf_file is not None
    saver.log_info(f'Found {gexf_file} matching kernel name {kernel_name}')
    return get_data_list(gexf_files=[gexf_file], gexf_folder='N/A', db_paths_override=[db_file_path],
                         to_save=False)


def get_data_list(gexf_files=GEXF_FILES, gexf_folder=GEXF_FOLDER, db_paths_override=None, to_save=True):
    # base_csv = pd.read_csv(join(get_root_path(), 'dse_database', 'databases', 'base.csv'))
    # name_cycle_map = dict(zip(base_csv.Kernel_name, base_csv.CYCLE))
    saver.log_info(f'Destination SAVE_DIR={SAVE_DIR}')
    saver.log_info(f'Found {len(gexf_files)} gexf files under {gexf_folder}')
    database = create_database()

    if FLAGS.model == 'code2vec':
        preprocessors = init_preprocessors_code2vec()
    else:
        assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            preprocessors = init_preprocessors_multi_modality()
        else:
            if FLAGS.sequence_modeling:
                preprocessors = init_preprocessors_src_code()
            else:
                preprocessors = init_preprocessors_programl()

    data_list = []

    all_gs = OrderedDict()

    # X_ntype_all = []
    # X_ptype_all = []
    # X_itype_all = []
    # X_ftype_all = []
    # X_btype_all = []
    #
    # edge_ftype_all = []
    # edge_ptype_all = []
    tot_configs = 0
    num_files = 0

    init_feat_dict = {}
    max_pragma_length = 0
    if FLAGS.pragma_encoder:
        for gexf_file in tqdm(GEXF_FILES[0:]):
            saver.log_info(f'now processing {gexf_file}')
            db_paths = []
            # n = basename(gexf_file).split('_')[0]
            _, n = _get_gname_n(gexf_file)
            for db_p in db_path:
                if FLAGS.v2_db:
                    if FLAGS.check_release_db:
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                            '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v18' not in f and 'v20' in f]  # and not 'updated' in f
                    else:
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                            '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v18' not in f and 'v20' in f and f'one-db-extended-round{FLAGS.round_num}' in f]  # and not 'updated' in f
                else:
                    if FLAGS.check_release_db:
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                            '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' not in f and 'v18' in f]  # and not 'updated' in f
                    else:
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                            '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' not in f and 'v18' in f and f'one-db-extended-round{FLAGS.round_num}' in f]  # and not 'updated' in f
                db_paths.extend(paths)
            if db_paths is None:
                saver.warning(f'No database found for {n}. Skipping.')
                continue
            database.flushdb()
            for idx, file in enumerate(db_paths):
                saver.log_info(f'processing db_paths for {n}: {file}')
                with open(file, 'rb') as f_db:
                    database.hmset(0, pickle.load(f_db))
                break
            keys = [k.decode('utf-8') for k in database.hkeys(0)]
            for key in sorted(keys):
                obj = pickle.loads(database.hget(0, key))
                # try:
                if type(obj) is int or type(obj) is dict:
                    continue
                if FLAGS.task == 'regression' and key[0:3] == 'lv1':  # or obj.perf == 0:#obj.ret_code.name == 'PASS':
                    continue
                if FLAGS.task == 'regression' and not FLAGS.invalid and obj.perf == 0:
                    continue
                #### TODO !! fix databases that have this problem:
                if obj.point == {}:
                    continue
                len_pragmas = len(obj.point)
                max_pragma_length = max(max_pragma_length, len_pragmas)
                break
    else:
        max_pragma_length = 21
    saver.log_info(f'max_pragma_length={max_pragma_length}')

    all_data_df = defaultdict(list)
    debug_cnt = 0  # just for debugging
    for gexf_file in tqdm(GEXF_FILES[0:]):  # TODO: change for partial/full data
        # if '2mm' in gexf_file or 'bicg' in gexf_file:
        #     continue
        # if '2mm' not in gexf_file:
        #     continue
        saver.info(f'Working on graph file: {gexf_file}')
        if FLAGS.dataset == 'vitis-cnn':
            if FLAGS.task == 'regression' and FLAGS.tag == 'only-vitis' and 'cnn' in gexf_file:
                continue
            # if 'cnn' not in gexf_file:  # TODO: change back for normal dataset
            #     continue
            pass
        elif FLAGS.dataset == 'simple-programl':
            pass
        elif FLAGS.dataset == 'machsuite' or 'programl' in FLAGS.dataset:
            proceed = False
            for k in ALL_KERNEL:
                to_check = f'{k}_'
                if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
                    to_check = f'{k}'
                if to_check in gexf_file:
                    proceed = True
                    break
            if not proceed:
                saver.info(f'Skipping this file as the kernel name is not selected. Check config file.')
                continue
            # pass
        else:
            raise NotImplementedError()

        if FLAGS.dataset == 'simple-programl':
            gname = basename(dirname(gexf_file))
            # n = basename(dirname(gexf_file))
        else:
            gname = basename(gexf_file).split('.')[0]
            # n = basename(gexf_file).split('_')[0]

        print(f'Reading gexf {gexf_file}')

        if hasattr(FLAGS, 'ignore_kernels') and len(FLAGS.ignore_kernels) > 0:
            if gname.split('_processed')[0] in FLAGS.ignore_kernels:
                saver.log_info(f'Ignoring {gname} due to FLAGS.ignore_kernels={FLAGS.ignore_kernels}')
                continue

        # if 'aes' not in gexf_file:
        #     continue
        if FLAGS.model == 'code2vec':
            g = read_code2vec_data(gexf_file)
        else:
            assert FLAGS.model == 'our'
            if FLAGS.multi_modality:
                g = read_multi_modality_data(gexf_file)
            else:
                if FLAGS.sequence_modeling:
                    g = read_source_code(gexf_file)
                    if g is None:
                        raise ValueError(f'Cannot load the graph for {gexf_file}; skip')
                        # continue
                else:
                    g = read_programl_graph(gexf_file)

        g.variants = OrderedDict()

        saver.log_info(gname)
        all_gs[gname] = g

        gname, n = _get_gname_n(gexf_file)
        saver.log_info(gname)

        # db_path = f'./all_dbs/{n}_result.db'
        if FLAGS.dataset == 'vitis-cnn':
            if n != 'cnn1':
                db_paths = glob(f'../{dse_database_name}/databases/vitis/exhaustive/{n}_result.db')
                # if FLAGS.task != 'regression':
                db_paths += glob(f'../{dse_database_name}/databases/vitis/bottleneck/{n}_result.db')
            else:
                db_paths = glob(f'../{dse_database_name}/databases/cnn_case1/{n}_result*.db')
        elif FLAGS.dataset == 'machsuite':
            db_paths = glob(f'../{dse_database_name}/machsuite/databases/exhaustive/{n}_result*.db')
            db_paths += glob(f'../{dse_database_name}/machsuite/databases/bottleneck/{n}_result*.db')
        elif FLAGS.dataset == 'simple-programl':
            db_paths = [f for f in
                        iglob(join(get_root_path(), f'{dse_database_name}/simple-program/databases/**'), recursive=True) if
                        f.endswith('.db') and n in f and 'one-db' in f]
        elif FLAGS.dataset == 'programl':
            db_paths = []
            for db_p in db_path:
                if FLAGS.v2_db:
                    if FLAGS.only_common_db:
                        if FLAGS.test_extra:
                            paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                                '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'extra' in f]  # and not 'updated' in f
                        else:
                            paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                                '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'common-dse4' in f]  # and not 'updated' in f
                    else:
                        if FLAGS.only_new_points:
                            paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                                'extra_3.db') and n in f and 'large-size' not in f and not 'archive' in f and 'extra' in f]  # and not 'updated' in f
                        else:
                            if FLAGS.check_release_db:
                                paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                                    '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' in f and 'v18' not in f and 'test' not in f]
                            else:
                                # paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' in f and 'corrupt' not in f]
                                paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                                    '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' in f and 'v18' not in f and 'one-db' in f and 'test' not in f and f'extended-round{FLAGS.round_num}' in f]
                else:
                    if FLAGS.check_release_db:
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                            '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' not in f and 'v18' in f]
                    else:
                        tag = 'one-db'
                        if FLAGS.FT_extra: tag = 'extra'
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                            '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' not in f and 'v18' in f and f'{tag}-extended-round{FLAGS.round_num}' in f]  # and 'extended' not in f and 'round' not in f # and 'gae-on' in f] # and not 'updated' in f
                db_paths.extend(paths)
            if db_paths is None:
                saver.warning(f'No database found for {n}. Skipping.')
                continue
        elif FLAGS.dataset == 'programl-machsuite':
            # db_paths_dict = {}
            # for KERNEL in MACHSUITE_KERNEL:
            db_paths = [f for f in iglob(db_path, recursive=True) if f.endswith(
                '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' not in f]
            #    db_paths_dict[KERNEL] = db_paths
        else:
            raise NotImplementedError()

        # db_paths = sorted(db_paths)
        database.flushdb()
        saver.log_info(f'db_paths for {n}:')
        for d in db_paths:
            saver.log_info(f'{d}')
        if len(db_paths) == 0:
            saver.log_info(f'{n} has no db_paths')

        if len(db_path) == 0:
            saver.warning(f'no database file for {n}')
            continue

        # if FLAGS.v2_db or FLAGS.FT_extra: # and (FLAGS.test_extra or FLAGS.only_new_points):
        #     if len(db_path) == 0:
        #         saver.warning(f'no database file for {n}')
        #         continue
        # else:
        #     assert len(db_paths) >= 1

        # load the database and get the keys
        # the key for each entry shows the value of each of the pragmas in the source file
        for idx, file in enumerate(db_paths):
            f_db = open(file, 'rb')
            # print('loading', f_db)
            data = pickle.load(f_db)
            database.hmset(0, data)
            max_idx = idx + 1
            f_db.close()
        # data = pickle.load(f_db)
        # database.hmset(0, mapping=data)
        keys = [k.decode('utf-8') for k in database.hkeys(0)]
        lv2_keys = [k for k in keys if 'lv2' in k]
        saver.log_info(f'num keys for {n}: {len(keys)} and lv2 keys: {len(lv2_keys)}')

        got_reference = False
        res_reference = 0
        max_perf = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            if key[0:3] == 'lv1' or obj.perf == 0:  # obj.ret_code.name == 'PASS':
                continue
            if obj.perf > max_perf:
                max_perf = obj.perf
                got_reference = True
                res_reference = obj
        if res_reference != 0:
            saver.log_info(f'reference point for {n} is {res_reference.perf}')
        else:
            saver.log_info(f'did not find reference point for {n} with {len(keys)} points')

        cnt = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            if FLAGS.task == 'regression' and key[0:3] == 'lv1':  # or obj.perf == 0:#obj.ret_code.name == 'PASS':
                continue
            if FLAGS.task == 'regression' and not FLAGS.invalid and obj.perf < FLAGS.min_allowed_latency:
                continue
            #### TODO !! fix databases that have this problem:
            if obj.point == {}:
                # exit()
                continue
            cnt += 1
            # print(key, obj.point)
            if FLAGS.model == 'code2vec':
                xy_dict, edge_dict = encode_feat_dict_code2vec(g, preprocessors, point=obj.point)
            else:
                assert FLAGS.model == 'our'
                if FLAGS.multi_modality:
                    xy_dict, edge_dict = encode_feat_dict_multi_modality(g, preprocessors, point=obj.point)
                else:
                    if FLAGS.sequence_modeling:
                        xy_dict, edge_dict = encode_feat_dict_src_code(g, preprocessors, point=obj.point)
                    else:
                        xy_dict, edge_dict = encode_feat_dict_programl(g, preprocessors, point=obj.point)

            pragmas = []
            pragma_name = []
            for name, value in sorted(obj.point.items()):
                if type(value) is str:
                    if value.lower() == 'flatten':
                        value = 2
                    elif value.lower() == 'off':
                        value = 1
                    elif value.lower() == '':
                        value = 3
                    else:
                        raise ValueError()
                elif type(value) is int:
                    pass
                else:
                    raise ValueError()
                pragmas.append(value)
                pragma_name.append(name)

            # if 'gemver' in gname:
            #     print(len(pragmas), obj.point)
            if 'gemver' in gname and len(pragmas) == 21:
                database.hdel(0, key)
                saver.warning(f'deleted {key} from database of {gname}')
                assert len(db_paths) == 1
                persist(database, db_paths[0])
                continue

            check_dim = init_feat_dict.get(gname)
            if check_dim is not None:
                # saver.info((gname, check_dim, len(pragmas)))
                assert check_dim[0] == len(pragmas), print(check_dim, len(pragmas))
                # if check_dim == len(pragmas):
                #     pass
                # else:
                #     database.hdel(0, key)
                #     print(check_dim, len(pragmas))
                #     saver.warning(f'deleted {key} from database of {gname}')
                #     assert len(db_paths) == 1
                #     persist(database, db_paths[0])
                #     continue
            else:
                init_feat_dict[gname] = [len(pragmas)]
            if FLAGS.pragma_uniform_encoder:
                pragmas.extend([0] * (max_pragma_length - len(pragmas)))

            xy_dict['pragmas'] = torch.FloatTensor(np.array([pragmas]))

            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(obj.perf + FLAGS.epsilon)
                        elif 'const' in FLAGS.norm_method:
                            y = obj.perf * FLAGS.normalizer
                            if y == 0:
                                y = FLAGS.max_number * FLAGS.normalizer
                            if FLAGS.norm_method == 'const-log2':
                                y = math.log2(y)
                        elif 'speedup' in FLAGS.norm_method:
                            assert obj.perf != 0
                            # assert got_reference == True
                            y = FLAGS.normalizer / obj.perf
                            # y = res_reference.perf / obj.perf
                            if FLAGS.norm_method == 'speedup-log2':
                                y = math.log2(y) / 2
                        elif FLAGS.norm_method == 'off':
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                        xy_dict['kernel_speedup'] = torch.FloatTensor(
                            np.array([math.log2(res_reference.perf / obj.perf)]))

                    elif tname == 'quality':
                        y = finte_diff_as_quality(obj, res_reference)
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(y + FLAGS.epsilon)
                        elif FLAGS.norm_method == 'const':
                            y = y * FLAGS.normalizer
                        elif FLAGS.norm_method == 'off':
                            pass
                    elif 'util' in tname or 'total' in tname:
                        y = obj.res_util[tname] * FLAGS.util_normalizer
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            elif FLAGS.task == 'class':
                # print(key, type(key))
                key = str(key)
                if 'lv1' in key:
                    lv2_key = key.replace('lv1', 'lv2')
                    if lv2_key in keys:
                        continue
                    else:
                        y = 0
                else:
                    y = 0 if obj.perf < FLAGS.min_allowed_latency else 1
                xy_dict['perf'] = torch.FloatTensor(np.array([y])).type(torch.LongTensor)
            else:
                raise NotImplementedError()

            vname = key

            if FLAGS.subtask == 'train' and FLAGS.MAML and FLAGS.MAML_split_kernel and KEY_PRAGMA in obj.point:
                xy_dict[KEY_PRAGMA] = obj.point[KEY_PRAGMA]
            xy_dict['point'] = str(obj.point)
            g.variants[vname] = (xy_dict, edge_dict)

            # if _need_fit_feature_encoder():
            #     X_ntype_all += xy_dict['X_ntype']
            #     X_ptype_all += xy_dict.get('X_ptype', [])
            #     X_itype_all += xy_dict.get('X_itype', [])
            #     X_ftype_all += xy_dict.get('X_ftype', [])
            #     X_btype_all += xy_dict.get('X_btype', [])
            #
            # edge_ftype_all += edge_dict['X_ftype']
            # edge_ptype_all += edge_dict['X_ptype']

            if FLAGS.model == 'our' and not FLAGS.sequence_modeling or (
                    FLAGS.sequence_modeling and FLAGS.data_repr == 'ast'):
                X_pragma_dict_repr = xy_dict['X_pragma_dict_repr']
                df_data = {**X_pragma_dict_repr}
                for tname in TARGETS:
                    if tname in df_data:
                        df_data[tname] = xy_dict[tname].item()
                all_data_df[gname].append(df_data)
                xy_dict.pop('X_pragma_dict_repr')  # otherwise will have key error in data loader

            debugging = False
            # debugging = True  # be careful
            if debugging:
                print('Debugging! Skip other designs in this kernel')
                break  # TODO: uncomment for debugging
                # if tot_configs >= 3:
                #     print('Debugging! Skip other designs in this kernel; Only allow at most 3 designs in total')
                #     break  # TODO: uncomment for debugging

        saver.log_info(f'final valid: {cnt}')
        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')

        for cname, counter in preprocessors.get('counters', {}).items():
            if cname != 'num_pragmas' and cname != 'num_tokens':
                saver.log_info(f'\t{cname} {len(counter)} {counter}')

        # save all_data_df to disk
        # for gname, data_li in all_data_df.items():
        #     fn = join(saver.get_log_dir(), f'{gname.split("_")[0]}_labeled_data.csv')
        #     pd.DataFrame.from_records(data_li).to_csv(fn)
        #     saver.log_info(f'Saved csv to {fn}')

        debugging = False
        # debugging = True # be careful
        if debugging:
            debug_cnt += 1
            if debug_cnt == 3:
                print('Debugging! Skip other kernels; only has 3 kernels in total')
                break  # TODO: uncomment for debugging

    # itype_vocab = {}
    # if not FLAGS.sequence_modeling:
    #     enc_ntype.fit(X_ntype_all)
    #     enc_ptype.fit(X_ptype_all)
    #     if FLAGS.itype_mask_perc > 0:
    #         X_itype_all += [[MASK_TOKEN]]
    #     enc_itype.fit(X_itype_all)
    #     for itype in X_itype_all:
    #         assert len(itype) == 1
    #         itype = itype[0]
    #         if itype not in itype_vocab:
    #             itype_vocab[itype] = len(itype_vocab) + 1
    #     enc_ftype.fit(X_ftype_all)
    #     enc_btype.fit(X_btype_all)
    #
    #     if len(edge_ftype_all) != 0:
    #         enc_ftype_edge.fit(edge_ftype_all)
    #     if len(edge_ptype_all) != 0:
    #         enc_ptype_edge.fit(edge_ptype_all)

    scalers = {}
    if FLAGS.target_preproc != 'None':
        saver.log_info('minmax_scaling...')
        ys_dict = defaultdict(list)
        for gname, g in tqdm(all_gs.items()):
            for vname, (d_node, _, _) in g.variants.items():
                for tname in TARGETS:
                    y = d_node[tname].item()
                    ys_dict[tname].append(y)
        ys_dict_trans = {}
        for tname in TARGETS:
            if FLAGS.target_preproc == 'minmax':
                scaler = MinMaxScaler()
            elif FLAGS.target_preproc == 'z-score':
                scaler = StandardScaler()
            else:
                raise ValueError()
            y_li = np.array(ys_dict[tname]).reshape(-1, 1)
            scaler.fit(y_li)
            if FLAGS.target_preproc == 'minmax':
                saver.log_info(f'scalar for {tname} range: {scaler.data_min_} {scaler.data_max_} '
                               f'with {scaler.n_samples_seen_} examples')
            elif FLAGS.target_preproc == 'z-score':
                saver.log_info(f'scalar for {tname} mean: {scaler.mean_} std {scaler.var_} '
                               f'with {scaler.n_samples_seen_} examples')
            else:
                raise ValueError()
            ys_dict_trans[tname] = scaler.transform(y_li)
            scalers[tname] = scaler
        i = 0
        for gname, (g, _) in tqdm(all_gs.items()):
            for vname, (d_node, _, _) in g.variants.items():
                for tname in TARGETS:
                    d_node[tname] = torch.FloatTensor(np.array([ys_dict_trans[tname][i]]))
                i += 1
        saver.log_info('minmax_scaling done')

    saver.log_info(f'Done {num_files} files tot_configs {tot_configs}')
    saver.log_info(f'GEXF_FILES={GEXF_FILES} (len={len(GEXF_FILES)})')

    seen_pragma_trans_types = set()
    # if not FLAGS.sequence_modeling:

    if FLAGS.model == 'code2vec':
        fit_preprocessors_code2vec(preprocessors)
    else:
        assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            fit_preprocessors_multi_modality(preprocessors)
        else:
            if FLAGS.sequence_modeling:
                fit_preprocessors_src_code(preprocessors)
            else:
                fit_preprocessors_programl(preprocessors)

    saver.log_info(f'Sklearn encoders fitted')
    # for cname, counter in preprocessors.get('counters', {}).items():
        # saver.log_info(f'\t{cname} {len(counter)} {counter}')

    if FLAGS.ptrans:  # TODO: handle pragma "nodes" for sequence modeling
        # Fill in empty tensor for pragma transformation nids.
        for gname, g in tqdm(all_gs.items()):
            for vname, d in g.variants.items():
                d_node, d_edge = d
                seen_pragma_trans_types.update(d_node['X_pragma_trans'].keys())
        saver.log_info(
            f'Found {len(seen_pragma_trans_types)} pragma transformation module types: {seen_pragma_trans_types}')
        for gname, g in tqdm(all_gs.items()):
            num_nodes = g.number_of_nodes()
            for vname, d in g.variants.items():
                d_node, d_edge = d
                for tstype in seen_pragma_trans_types:
                    cur_nids = d_node['X_pragma_trans'].get(tstype)
                    new_nids = torch.zeros(num_nodes)
                    if cur_nids is not None:
                        new_nids[cur_nids] = 1
                    d_node['X_pragma_trans'][tstype] = new_nids

    saver.log_info('Start encoding gs')
    for gname, g in tqdm(all_gs.items()):
        # edge_index = create_edge_index_both(g)
        saver.log_info('edge_index created', gname)
        new_gname = gname.split('_')[0]
        if FLAGS.subtask == 'train' and FLAGS.MAML and FLAGS.MAML_split_kernel and KEY_PRAGMA in d_node:
            new_gname = gname.split('_')[0] + '-' + d_node[KEY_PRAGMA]

        for vname, d in g.variants.items():
            d_node, d_edge = d

            if FLAGS.model == 'code2vec':
                data = encode_X_torch_code2vec(g, d_node, d_edge, preprocessors, new_gname, vname)
            else:
                assert FLAGS.model == 'our'
                if FLAGS.multi_modality:
                    data = encode_X_torch_multi_modality(g, d_node, d_edge, preprocessors, new_gname, vname)
                else:
                    if FLAGS.sequence_modeling:
                        data = encode_X_torch_src_code(d_node, d_edge, preprocessors, new_gname, vname)
                    else:
                        data = encode_X_torch_programl(g, d_node, d_edge, preprocessors, new_gname, vname)

            if FLAGS.task == 'regression':
                data_list.append(data)
            elif FLAGS.task == 'class':
                data_list.append(data)
            else:
                raise NotImplementedError()

    if FLAGS.model == 'code2vec':
        from data_code2vec import not_found_cnt
        saver.log_info(f'Done loading code2vec data (i.e. pretrained design embeddings); not_found_cnt={not_found_cnt}')
    else:
        assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
                from data_multi_modality import PC_LINKS_DICT
                save_pickle(PC_LINKS_DICT, join(saver.get_obj_dir(), 'pc_links_dict.pkl'))

                print_stats(preprocessors['preprocessors_src_code']['counters']['num_matched_pairs'],
                            'number of matched node-token pairs per program')
        else:
            if FLAGS.sequence_modeling:
                nns = [len(d.x) for d in data_list]
                print_stats(nns, 'number of chunks for pure seq or number of nodes for ast')
                from data_src_code import FUNCTION_NAMES, VARIABLE_NAMES
                saver.log_info(
                    f'FUNCTION_NAMES={FUNCTION_NAMES}')  # so that we can enrich the tokenizer with these additional names
                saver.log_info(f'VARIABLE_NAMES={VARIABLE_NAMES}')

                print_stats(preprocessors['counters']['num_tokens'], 'number of tokens per program')

            else:
                nns = [d.x.shape[0] for d in data_list]
                print_stats(nns, 'number of nodes')

                from data_programl import diameters_list
                print_stats(diameters_list, 'graph diameter')

        # exit()

        if not FLAGS.sequence_modeling:
            if not FLAGS.only_pragma_nodes:
                nnodes = [d.x.shape[0] for d in data_list]
                print_stats(nnodes, 'num nodes')

                nedges = [d.edge_index.shape[1] for d in data_list]
                print_stats(nedges, 'num edges')

                ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
                print_stats(ads, 'avg degrees')

                print_stats(preprocessors['counters']['num_pragmas'], 'number of pragmas per program')

    # for d in data_list:
    #     print(d.num_features)

    # ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
    # print_stats(ads, 'avg degrees')
    saver.info(data_list[0])

    saver.log_info(f'dataset[0].num_features {data_list[0].num_features}')
    for target in TARGETS:
        if not hasattr(data_list[0], target.replace('-', '_')):
            saver.warning(f'Data does not have attribute {target}')
            continue
        ys = [_get_y(d, target).item() for d in data_list]
        # if target == 'quality':
        #     continue
        plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True,
                  bins=None)
        if FLAGS.task == 'class':
            saver.log_info(f'{target}_ys: {Counter(ys)}')

    # from sequence_modeling import CODE_LEN_LI
    # print_stats(CODE_LEN_LI, 'CODE_LEN_LI')

    if FLAGS.force_regen and to_save:
        saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
        rmtree(SAVE_DIR)
        create_dir_if_not_exists(SAVE_DIR)
        for i in tqdm(range(len(data_list))):
            torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))

    if FLAGS.force_regen and to_save:
        # if FLAGS.only_pragma:
        #     obj = {'enc_ptype': enc_ptype}
        # else:
        # obj = {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype,
        #        'enc_itype': enc_itype, 'enc_ftype': enc_ftype,
        #        'enc_btype': enc_btype,
        #        'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge,
        #        'itype_vocab': itype_vocab
        #        }

        # if FLAGS.model == 'code2vec':
        #     saver.log_info(f'code2vec: Skip encoder saving)')
        # else:
        # assert FLAGS.model == 'our'
        p = ENCODER_PATH
        if FLAGS.multi_modality:
            save({'encoders_programl': preprocessors['preprocessors_programl']['encoders'],
                  'encoders_src_code': preprocessors['preprocessors_src_code']['encoders']}, p)
        else:
            # if FLAGS.encoder_path == None:
            save(preprocessors['encoders'], p)

        if FLAGS.pragma_uniform_encoder:
            for gname in init_feat_dict:
                init_feat_dict[gname].append(max_pragma_length)
        name = 'pragma_dim'
        # if FLAGS.v2_db:
        #     save(obj_p, join(SAVE_DIR, name))
        #     name += '_v2'
        save(init_feat_dict, join(SAVE_DIR, name))

        for gname, feat_dim in init_feat_dict.items():
            saver.log_info(f'{gname} has initial dim {feat_dim[0]}')

    if not to_save:
        return data_list

    rtn = MyOwnDataset()

    rtn.add_target_scalers(scalers, to_save=to_save)
    rtn.add_attribute('seen_pragma_trans_types', sorted(list(seen_pragma_trans_types)))
    rtn.add_attribute('init_feat_dict', init_feat_dict)

    return rtn, init_feat_dict


def _get_gname_n(gexf_file):
    if FLAGS.dataset == 'simple-programl':
        # gname = basename(dirname(gexf_file))
        # n = basename(dirname(gexf_file))
        gname = basename(gexf_file).split('.')[0]
        n = f"{basename(gexf_file).split('_')[0]}_"
    else:
        gname = basename(gexf_file).split('.')[0]
        n = f"{basename(gexf_file).split('_')[0]}_"
    if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
        n = f"{basename(gexf_file).split('.')[0]}_"  # tricky: need the _ at the end!
        if n == 'stencil_stencil2d_':
            n = 'stencil_'
    return gname, n


# def _need_fit_feature_encoder():
#     if FLAGS.sequence_modeling:
#         if FLAGS.data_repr == 'ast':
#             return True
#         else:
#             return False  # pure sequence
#     else:
#         return True


def create_edge_index_both(g_programl):
    edge_index = create_edge_index(g_programl)

    return edge_index


def _get_y(data, target):
    return getattr(data, target.replace('-', '_'))


def print_data_stats(data_loader, tvt):
    nns, ads, ys = [], [], []
    for d in tqdm(data_loader):
        nns.append(d.x.shape[0])
        # ads.append(d.edge_index.shape[1] / d.x.shape[0])
        ys.append(d.y.item())
    print_stats(nns, f'{tvt} number of nodes')
    # print_stats(ads, f'{tvt} avg degrees')
    plot_dist(ys, f'{tvt} ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
    saver.log_info(f'{tvt} ys', Counter(ys))


def load_encoders():
    path = FLAGS.encoder_path
    if path == 'None' or path is None:
        # saver.log_info(f'encoder path is None; use default path instead')
        path = ENCODER_PATH
    rtn = load(path, print_msg=False)
    # saver.log_info(f'Loaded encoder from {path}')
    if rtn is None:
        raise ValueError(f'Trying to load encoders; Check encoder path {path}; Maybe force regen '
                         f'or check load_encoders path')
    return rtn


def _encode_fulltext(g, text_encoder):
    X_fulltext = []

    for node, ndata in sorted(g.nodes(data=True)):  # TODO: node ordering
        with torch.no_grad():
            s = ndata.get('full_text', '')
            if s == '':
                s = ndata.get('content', '')
            X = text_encoder.encode(s)
            assert X.shape == (1, text_encoder.dim())  # Note: X_fulltext i trainable

        X_fulltext.append(X)

    return torch.cat(X_fulltext, dim=0)


def _encode_edge_torch(edge_dict, enc_ftype, enc_ptype):
    """
    edge_dict is the dictionary returned by _encode_edge_dict
    """
    X_ftype = enc_ftype.transform(edge_dict['X_ftype'])
    X_ptype = enc_ptype.transform(edge_dict['X_ptype'])

    X = hstack((X_ftype, X_ptype))
    X = coo_to_sparse(X)
    X = X.to_dense()

    return X


def get_num_features(dataset, sequence_modeling=FLAGS.sequence_modeling):
    if sequence_modeling:
        from data_src_code import get_code_encoder_dim
        num_features = get_code_encoder_dim()
        if FLAGS.data_repr == 'ast' and FLAGS.AST_combine_node_edge_labels:
            return num_features + dataset[0].X_ast_node_labels.shape[1]
    else:
        num_features = dataset[0].num_features
    return num_features
