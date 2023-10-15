from train import train, create_optimizer
from test import test
from config import FLAGS
from saver import saver
from data import split_li, create_dataloader

from model import feature_extract
from model_factory import create_model

import torch
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import random
import numpy as np
import copy

PLUSMINUS_SYMBOL = u"\u00B1"


def adapt_main(dataset, pragma_dim):
    # The key characteristic of adaptation is that it is done per kernel,
    # i.e. for each new/unseen test kernel (inductive), adaptation is performed on it
    # by copying the model's weights and doing some adaptation.
    if FLAGS.tvt_split_by != 'kernels_inductive':
        saver.log_info(f'adapt_main: FLAGS.tvt_split_by != "kernels_inductive ({FLAGS.tvt_split_by})"')
        return

    dataloaders = _split_into_indiviual_dataloaders(dataset)

    eval_dict = OrderedDict()
    eval_dict['before_adaptation'] = OrderedDict()
    eval_dict['after_adaptation'] = OrderedDict()
    for gname, dls in dataloaders.items():
        for k in eval_dict.keys():
            eval_dict[k][gname] = []
        for j, dl in enumerate(dls):
            eval_dict = _adapt_and_test(dl, pragma_dim, f'{gname}_{j}', eval_dict, gname)

    for key, eval_d in eval_dict.items():
        saver.log_info(f'-' * 20)
        final_print_dict = OrderedDict()
        saver.log_info(f'{key}:')
        all_rmses = []
        for gname, rmse_list in eval_d.items():
            rmse_list_s = [f'{x:.4f}' for x in rmse_list]
            fps = _avg_plusminus_std_for_li(rmse_list)
            saver.log_info(f'\t{gname}: {rmse_list_s} --> {fps}')
            final_print_dict[gname] = fps
            all_rmses += rmse_list
        all_agg_key = f'all/aggregated({len(all_rmses)})'
        keys_to_use = [all_agg_key] + list(final_print_dict.keys())
        final_print_dict[all_agg_key] = _avg_plusminus_std_for_li(all_rmses)
        ks = "\t".join(keys_to_use)
        vs = "\t".join([final_print_dict[k] for k in keys_to_use])
        saver.log_info(f'Summary of adaptation for {key}:')
        saver.log_info(f'{ks}')
        saver.log_info(f'{vs}')
        saver.log_info(f'-' * 20)
    saver.log_info(f'Adaptation done')
    # exit()


def _avg_plusminus_std_for_li(li):
    avg = np.mean(li)
    std = np.std(li)
    return f'{avg:.4f}{PLUSMINUS_SYMBOL}{std:.2f}'


def _adapt_and_test(dl, pragma_dim, test_name, eval_dict, gname):
    model = create_model(dl.adaptation_dl, pragma_dim)

    if FLAGS.sequence_modeling and not FLAGS.finetune_bert:
        for name, param in model.bert_model.named_parameters():
            param.requires_grad = False
            saver.log_info(f'No fine tune bert: Freezing param {name}')

    if FLAGS.load_model != None and FLAGS.load_model != 'None':
        model, _ = saver.load_trained_model(FLAGS.load_model, model)

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)

    if len(eval_dict['before_adaptation'][gname]) == 0:  # otherwise, no need to test the model again and again
        r_before_adapt = _test(model, dl, f'{test_name}_before')
        eval_dict['before_adaptation'][gname].append(r_before_adapt)

    if FLAGS.num_mini_epochs >= 1:
        model = _adapt(model, dl)
        r_after_adapt = _test(model, dl, f'{test_name}_after')
        eval_dict['after_adaptation'][gname].append(r_after_adapt)

    return eval_dict


def _test(model, dl, test_name):
    testr, loss_dict, encode_loss, eval_df = test(dl.final_hold_out_dl, 'test', model, epoch=0, plot_test=True,
                                                  csv_dict=None,
                                                  data_dict=None, forward_pairwise=False,
                                                  eval_pairwise=False, test_name=f'{test_name}_holdout')

    # saver.log_info(f'testr={testr}')
    # saver.log_info(f'loss_dict={loss_dict}')
    # saver.log_info(f'encode_loss={encode_loss}')
    # print('@@@', eval_df)
    # exit(-1)
    if FLAGS.task == 'regression':
        assert eval_df.iloc[-1]['target'] == 'tot/avg'
        assert FLAGS.task == 'regression', f'Other task has not been implemented {FLAGS.task}'
        rmse = eval_df.iloc[-1]['rmse']
        rtn = rmse
    elif FLAGS.task == 'class':
        accuracy = eval_df['accuracy']
        assert 0 <= accuracy <= 1
        rtn = accuracy
    else:
        raise NotImplementedError()


    return rtn


def _adapt(model, dl):
    optimizer = create_optimizer(model)
    assert FLAGS.num_mini_epochs >= 1
    train_losses = []
    model_best = None
    for epoch in range(FLAGS.num_mini_epochs):
        loss, loss_dict_train, gae_loss_train, num_data = train(epoch, model, dl.adaptation_dl, optimizer,
                                                                forward_pairwise=False)
        if FLAGS.test_which_adapted_model == 'best_train' and (model_best is None or (train_losses and loss < min(train_losses))):
            print('FLAGS.test_which_adapted_model', FLAGS.test_which_adapted_model)
            model_best = copy.deepcopy(model)
        train_losses.append(loss)
        saver.log_info(f'Mini epoch {epoch}: loss={loss:.4f} num_data={num_data}')
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses))}')
    if FLAGS.test_which_adapted_model == 'last_epoch':
        pass
    elif FLAGS.test_which_adapted_model == 'best_train':
        assert model_best is not None
        model = model_best  # return the best model in training
        saver.log_info(f'Returning the best model according to train loss')
    else:
        raise NotImplementedError()
    return model


class AdaptationDataLoaders(object):
    def __init__(self, final_hold_out_dl, adaptation_dl):
        self.final_hold_out_dl = final_hold_out_dl
        self.adaptation_dl = adaptation_dl


def _split_into_indiviual_dataloaders(dataset):
    saver.log_info(f'Split the dataset by invidual kernels according to {FLAGS.test_kernels}')
    # train_file_li = []
    test_kernel_li_map = defaultdict(list)
    file_li = dataset.processed_file_names
    for file in tqdm(file_li):
        data = torch.load(file)
        if data.gname in FLAGS.test_kernels:
            test_kernel_li_map[data.gname].append(file)
            # train_file_li.append(file)
    saver.log_info(f'-' * 20)

    # saver.log_info(f'Found {len(train_file_li)} files for training')

    def _shuffle_split_li_into_2_chunks(li_, ratio=None, num=None, force_shuffle=False):

        if FLAGS.shuffle or force_shuffle:
            random.Random(123).shuffle(li_)
        if ratio is not None:
            r1 = int(len(li_) * ratio)
        elif num is not None:
            r1 = num
        else:
            assert False
        lengths = [r1, len(li_) - r1]
        rtn = split_li(li_, lengths)
        return rtn

    dataloaders = OrderedDict()
    for gname, test_file_li in sorted(test_kernel_li_map.items()):
        li = _shuffle_split_li_into_2_chunks(test_file_li, ratio=FLAGS.test_holdout_ratio)
        saver.log_info(
            f'\tval/test: \t{gname} has {len(test_file_li)} designs in total; split into others {len(li[1])} and final test {len(li[0])}')
        final_hold_out_data_points = li[0]
        final_hold_out_dl = create_dataloader(final_hold_out_data_points, shuffle=False)

        assert FLAGS.repeat_times >= 1
        rest_for_choosing = li[1]
        saver.log_info(f'Samoling {FLAGS.adaptation_num_dp} out of {len(rest_for_choosing)} for {FLAGS.repeat_times} '
                       f'times to get adaptation data points')
        assert len(rest_for_choosing) >= FLAGS.adaptation_num_dp, \
            f'Not enough rest designs to choose {len(rest_for_choosing)} < {FLAGS.adaptation_num_dp}'
        dataloaders[gname] = []
        for i in range(FLAGS.repeat_times):
            li = _shuffle_split_li_into_2_chunks(rest_for_choosing, num=FLAGS.adaptation_num_dp, force_shuffle=True)
            adaptation_data_points = li[0]
            assert len(adaptation_data_points) == FLAGS.adaptation_num_dp
            adaptation_dl = create_dataloader(adaptation_data_points, shuffle=FLAGS.shuffle)

            dataloaders[gname].append(AdaptationDataLoaders(final_hold_out_dl, adaptation_dl))

    return dataloaders
