from data import SAVE_DIR, MyOwnDataset, split_dataset

from config import FLAGS
from saver import saver
from utils import create_dir_if_not_exists, create_pred_dict

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import math
import random
from tqdm import tqdm
import os
from os.path import join
import torch
from shutil import rmtree
from glob import glob
from collections import defaultdict, OrderedDict
import pandas as pd
from scipy.spatial import distance


def get_pairwise_data_loaders(dataset, torch_generator, data_dict=None):
    # assert not FLAGS.sequence_modeling
    if FLAGS.force_regen_pairwise_data:
        # if FLAGS.split_pairs_by_holding_out == 'pairs':
        #     all_pair_li = []
        #     for gname, pair_li in pair_dict.items():
        #         all_pair_li += pair_li
        #
        #     train_li, val_li, test_li = _shuffle_a_list_split_into_3_tvt_chunks(all_pair_li, 'pairs')

        # elif FLAGS.split_pairs_by_holding_out == 'designs':

        train_ds, val_ds, test_ds, _ = split_dataset(dataset, torch_generator)
        train_design_li, val_design_li, test_design_li = set(), set(), set()

        for dl, tvt, design_li in [(train_ds, 'train', train_design_li), (val_ds, 'val', val_design_li),
                                   (test_ds, 'test', test_design_li)]:
            for d in dl:
                gnames = d.gname
                points = d.xy_dict_programl['point']
                for x, y in zip(gnames, points):
                    design_li.add((x, y))
                # design_li += d.xy_dict_programl['point']
            saver.log_info(
                f'{tvt} design_li: {len(design_li)}')
            assert len(design_li) != 0

        _check_overlapping(train_design_li, val_design_li, 'train_design_li', 'val_design_li')
        _check_overlapping(train_design_li, test_design_li, 'train_design_li', 'test_design_li')
        _check_overlapping(val_design_li, test_design_li, 'val_design_li', 'test_design_li')

        if data_dict is None:
            data_dict = get_data_dict_by_gname(dataset)
        pair_dict, *_ = gather_eval_pair_data(data_dict)

        num_pairs_in_total = 0
        for gname, pair_li in pair_dict.items():
            for pair in pair_li:
                data1, data2, _, _ = pair
                d1_str = data1.xy_dict_programl['point']
                d2_str = data2.xy_dict_programl['point']
                assert d1_str != d2_str
                num_pairs_in_total += 1
        if num_pairs_in_total == 0:
            raise ValueError(f'num_pairs_in_total=0')

        # train_design_li, val_design_li, test_design_li = _shuffle_a_list_split_into_3_tvt_chunks(
        #     all_designs, 'individual designs')

        train_li, val_li, test_li = [], [], []
        for gname, pair_li in pair_dict.items():
            for pair in pair_li:
                data1, data2, _, _ = pair
                d1_str = data1.xy_dict_programl['point']
                d2_str = data2.xy_dict_programl['point']
                assert d1_str != d2_str
                if (gname, d1_str) in train_design_li and (gname, d2_str) in train_design_li:
                    train_li.append(pair)
                if (gname, d1_str) in val_design_li and (gname, d2_str) in val_design_li:
                    val_li.append(pair)
                if (gname, d1_str) in test_design_li and (gname, d2_str) in test_design_li:
                    test_li.append(pair)
        for li, li_name in [(train_li, 'train_design_li'), (val_li, 'val_design_li'), (test_li, 'test_design_li')]:
            saver.log_info(
                f'Pairs (both data1 and data2) that are in {li_name}: {len(li)} (/{num_pairs_in_total}={len(li) / num_pairs_in_total:.4%})')
            assert len(li) != 0

        # else:
        #     assert False

        train_data_li = list(_get_pairwise_data_gen(train_li, 'train'))
        val_data_li = list(_get_pairwise_data_gen(val_li, 'val'))
        test_data_li = list(_get_pairwise_data_gen(test_li, 'test'))

    else:
        train_data_li, val_data_li, test_data_li = None, None, None

    train_loader = _get_data_loader(train_data_li, 'train', torch_generator)
    val_loader = _get_data_loader(val_data_li, 'val', torch_generator)
    test_loader = _get_data_loader(test_data_li, 'test', torch_generator)

    return train_loader, val_loader, test_loader


def _check_overlapping(a, b, label1, label2):
    num_dup = len(set(a) & set(b))
    saver.log_info(f'{label1} ({len(a)}) and {label2} ({len(b)}) have {num_dup} overlapping/intersection')


# def _shuffle_a_list_split_into_3_tvt_chunks(li, label):
#     random.Random(123).shuffle(li)
#
#     num_data = len(li)
#     saver.log_info(f'Found {num_data} {label}  (shuffled)')
#
#     r1 = int(num_data * (1.0 - 2 * (FLAGS.val_ratio)))
#     r2 = int(num_data * (FLAGS.val_ratio))
#     train_li = li[0:r1]
#     val_li = li[r1:r1 + r2]
#     test_li = li[r1 + r2:]
#     saver.log_info('Split into 3 chunks:')
#     saver.log_info(f'\ttrain_li: {len(train_li)} (/{num_data}={len(train_li) / num_data:.4%})')
#     saver.log_info(f'\tval_li: {len(val_li)} (/{num_data}={len(val_li) / num_data:.4%})')
#     saver.log_info(f'\ttest_li: {len(test_li)} (/{num_data}={len(test_li) / num_data:.4%})')
#
#     return train_li, val_li, test_li


def _get_data_loader(batch_data_li, tvt, torch_generator):
    sp_s = '_split_d'

    tvt_split_s = ''
    if FLAGS.tvt_split_by == 'designs_transductive':
        if FLAGS.val_ratio != 0.15:
            tvt_split_s = f'_{FLAGS.val_ratio}'
    elif FLAGS.tvt_split_by == 'kernels_inductive':
        tvt_split_s = f'ind_{"_".join(FLAGS.test_kernels)}_{FLAGS.val_ratio_in_test_kernels}_{FLAGS.val_ratio_in_train_kernels}_{FLAGS.shuffle}'
    else:
        assert False

    save_dir = join(SAVE_DIR, f'pairwise_{tvt}_bs={FLAGS.batch_size}{sp_s}{tvt_split_s}')  # always save to disk first

    if FLAGS.force_regen_pairwise_data:
        saver.log_info(f'Saving {len(batch_data_li)} batch objects to disk {save_dir}; Deleting existing files')
        file_li = []
        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            rmtree(save_dir)
        create_dir_if_not_exists(save_dir)
        for i, batch_data in enumerate(tqdm(batch_data_li)):
            f = join(save_dir, f'data_{i}.pt')
            torch.save(batch_data, f)
            file_li.append(f)
        # batch size below must be one since it is already a batch object containing FLAGS.batch_size data points (see above save_dir path)
    else:
        saver.log_info(f'Globbing from {save_dir}')
        file_li = glob(join(save_dir, '*.pt'))
        if len(file_li) == 0:
            raise ValueError(
                f'Error! Pairwise data folder is empty -- consider setting force_regen_pairwise_data to True\nsave_dir={save_dir}')
        else:
            saver.log_info(f'Found {len(file_li)} pt files under pairwise data dir {save_dir}')
    return DataLoader(MyOwnDataset(data_files=file_li, need_encoders=False, need_attribs=False), batch_size=1,
                      shuffle=False, pin_memory=True,
                      generator=torch_generator)


def _get_pairwise_data_gen(li, tvt):
    num_iters_total = math.ceil(len(li) / FLAGS.batch_size)
    saver.log_info(
        f'Pairwise data loader {tvt}: {len(li)} data points (// {FLAGS.batch_size} --> {num_iters_total} iters)')
    for iter_id in range(num_iters_total):
        begin = iter_id * FLAGS.batch_size
        end = min(begin + FLAGS.batch_size, len(li))
        chunk = li[begin:end]
        individual_data_li = []
        for p in chunk:
            individual_data_li.append(p[0])  # critical: first/top half --> d_1
        for p in chunk:
            individual_data_li.append(p[1])  # critical: second/buttom half --> d_2
        if iter_id != num_iters_total - 1:
            assert len(individual_data_li) == 2 * FLAGS.batch_size
        batch = Batch.from_data_list(individual_data_li)
        yield batch


def gather_eval_pair_data(data_dict, points_pred_by_gname=None, pairs_pred_by_gname=None, target_list=None,
                          test_name=None):
    # data_dict = defaultdict(list)
    # for i, data in enumerate(tqdm(dataset)):
    #     # if i == 10:
    #     #     break  # TODO: uncomment for debugging
    #     data_dict[data.gname].append(data)

    pair_dict = {}
    tlist_for_acc = ['all']
    pred_dict_by_target_global = create_pred_dict(tlist_for_acc, extra_entries=['name'])
    for gname, data_li in data_dict.items():
        seen_designs = set()
        # pred_dict_by_target_local = create_pred_dict(tlist_for_acc)
        pair_li = []
        for i, data1 in enumerate(data_li):
            for j, data2 in enumerate(data_li):
                assert data1.gname == data2.gname
                if i < j:  # only check triangle
                    d1 = data1.xy_dict_programl['point']
                    d2 = data2.xy_dict_programl['point']
                    if _check_dict_diff_by_one(eval(d1), eval(d2)):
                        if points_pred_by_gname:
                            assert not pairs_pred_by_gname
                            _add_empty_li_if_not_exsits(pred_dict_by_target_global['all'], 'emb_diff')

                            pred_1 = points_pred_by_gname[gname].get(d1)
                            pred_2 = points_pred_by_gname[gname].get(d2)

                            if pred_1 is not None and pred_2 is not None:
                                for t in target_list:
                                    pred_comp, true_comp = _pairwise_get_pred(pred_1, pred_2, data1, data2, t)
                                    # pred_dict_by_target_local['all']['pred'].append(pred_comp)
                                    # pred_dict_by_target_local['all']['true'].append(true_comp)
                                    pred_dict_by_target_global['all']['pred'].append(pred_comp)
                                    pred_dict_by_target_global['all']['true'].append(true_comp)
                                    pred_dict_by_target_global['all']['name'].append((gname, d1, d2))

                                emb_T_1 = pred_1['emb_T']
                                emb_T_2 = pred_2['emb_T']
                                emb_diff = distance.euclidean(emb_T_1, emb_T_2)

                                pred_dict_by_target_global['all']['emb_diff'].append(emb_diff)

                        elif pairs_pred_by_gname:
                            pred_1 = None
                            pred_2 = None

                            assert not points_pred_by_gname
                            pred_dict = pairs_pred_by_gname[gname].get((d1, d2))
                            if pred_dict is not None:
                                for t in target_list:
                                    pred_val = pred_dict[t]
                                    assert pred_val == 0 or pred_val == 1
                                    pred_comp = pred_val
                                    true_comp = _get_comp_result_data(data1, data2, t)
                                    pred_dict_by_target_global['all']['pred'].append(pred_comp)
                                    pred_dict_by_target_global['all']['true'].append(true_comp)
                                    pred_dict_by_target_global['all']['name'].append((gname, d1, d2))

                        else:
                            pred_1, pred_2 = None, None
                        pair_li.append((data1, data2, pred_1, pred_2))
                        seen_designs.add(i)
                        seen_designs.add(j)

        # if len(pred_dict_by_target_local) > 0:
        #     _report_class_result(pred_dict_by_target_local, f'{gname}_pairwise_pred_dict_by_target_local:')

        ll = len(pair_li)
        tot = len(data_li) * len(data_li)
        saver.log_info(f'{gname}: Found {ll} pairs out of {len(data_li)}*{len(data_li)}={tot} pairs'
                       f' ({ll}/{tot}={ll / tot:.2%})'
                       f' -- seen_designs {len(seen_designs)}/{len(data_li)}={len(seen_designs) / len(data_li):.2%}', silent=True)
        pair_dict[gname] = pair_li

        _save_pair_li_as_csv(gname, pair_li, target_list)

    if test_name is not None:
        saver.save_dict(pred_dict_by_target_global, f'{test_name}_pred_dict_by_target_global.pkl')

    return pair_dict, pred_dict_by_target_global


def get_data_dict_by_gname(dataset):
    data_dict = defaultdict(list)
    saver.log_info(f'Going through all {len(dataset)} data points which will take a while')
    for i, file in enumerate(tqdm(dataset.processed_file_names)):
        data = torch.load(file)
        # if i == 100:
        #     break  # TODO: uncomment for debugging
        data_dict[data.gname].append(data)

    # below: slow
    # for i, data in enumerate(tqdm(dataset)):  # takes a while; not too fast; need to load one by one
    #     # if i == 100:
    #     #     break  # TODO: uncomment for debugging
    #     data_dict[data.gname].append(data)
    assert len(data_dict) > 0
    return data_dict


def _check_dict_diff_by_one(d1, d2):
    diff_count = 0
    for k1, v1 in d1.items():
        if d2[k1] != v1:
            diff_count += 1
    if diff_count == 0:
        saver.log_info(f'Warning! d1 and d2 are identical: d1={d1}\nd2={d2}')
    return diff_count == 1


def _save_pair_li_as_csv(gname, pair_li, target_list=None):
    if target_list is None:
        from data import TARGETS
        target_list = TARGETS
    fn = join(saver.get_log_dir(), f'{gname.split("_")[0]}_pair_data.csv')
    record_li = []
    for (data1, data2, pred1, pred2) in pair_li:
        repr = OrderedDict()
        for did, data in [(1, data1), (2, data2)]:
            for k, v in eval(data.xy_dict_programl['point']).items():
                repr[f'{did}_{k}'] = v

        for t in target_list:
            for did, data in [(1, data1), (2, data2)]:
                repr[f'{did}_{t}_true'] = data.xy_dict_programl[t].item()

        for t in target_list:
            for did, pred in [(1, pred1), (2, pred2)]:
                if pred is not None:
                    pred_val = pred[t]
                else:
                    pred_val = None
                repr[f'{did}_{t}_pred'] = pred_val

        for t in target_list:
            if pred1 is not None and pred2 is not None:
                pred_comp, true_comp = _pairwise_get_pred(pred1, pred2, data1, data2, t)
                repr[f'{t} 1>=2? true'] = true_comp
                repr[f'{t} 1>=2? pred'] = pred_comp

        record_li.append(repr)
    pd.DataFrame.from_records(record_li).to_csv(fn)
    # saver.log_info(f'Saved csv to {fn}')


def _pairwise_get_pred(pred_1, pred_2, data1, data2, t):
    pred_comp = _get_comp_result(pred_1[t], pred_2[t])
    true_comp = _get_comp_result_data(data1, data2, t)
    return pred_comp, true_comp


def _get_comp_result_data(data1, data2, t):
    return _get_comp_result(data1.xy_dict_programl[t].item(), data2.xy_dict_programl[t].item())


def _get_comp_result(e1, e2):
    return 1 if e1 <= e2 else 0  # TODO: double check M or > is NOT messed up


def get_comp_result_tagret(y1, y2):
    return (y1 <= y2).long().flatten()  # TODO: consistent with above: 1 if <=


def _add_empty_li_if_not_exsits(d, key):
    assert type(d) is dict
    if key not in d:
        d[key] = []
