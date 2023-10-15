from train_pairwise import get_pairwise_data_loaders, gather_eval_pair_data

from config import FLAGS
from saver import saver
from utils import _get_y_with_target, create_pred_dict, print_stats, create_loss_dict_get_target_list, update_loss_dict
from data import split_dataset, torch_generator

from model import feature_extract
from model_factory import create_model

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix

import torch
from scipy.stats import rankdata, kendalltau
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from os.path import dirname, basename


def inference(dataset, pragma_dim):
    saver.log_info(f'Inference begins')
    train_loader, val_loader, test_loader, transductive_test_loader = split_dataset(dataset, torch_generator)

    model = create_model(test_loader, pragma_dim)
    saver.log_model_architecture(model)

    loaded_model_info = None
    if FLAGS.load_model != None and FLAGS.load_model != 'None':
        model, loaded_model_info = saver.load_trained_model(FLAGS.load_model, model)

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
    # print(model)
    # saver.log_model_architecture(model)

    li = [('train_loader', train_loader, False)]
    if transductive_test_loader is not None:
        li += [('transductive_test_loader', transductive_test_loader, False)]
    li += [('test_loader', test_loader, False)]
    data_dict = None
    eval_pairwise = False
    # if not FLAGS.sequence_modeling: # TODO: tune it
    #     data_dict = get_data_dict_by_gname(dataset)  # loads all graphs into memory for analysis
    #     eval_pairwise = True
    if FLAGS.pairwise_class and FLAGS.loss_components == 'both':
        pairwise_train_loader, _, pairwise_test_loader = get_pairwise_data_loaders(dataset, torch_generator,
                                                                                   data_dict=data_dict)
        li = [('pairwise_test_loader', pairwise_test_loader, True),
              ('pairwise_train_loader', pairwise_train_loader, True)] + li

    final_print_dict = OrderedDict()
    for test_name, loader, forward_pairwise in li:
        saver.log_info(f'-' * 100)
        saver.log_info(f'{test_name} starts')
        if FLAGS.task == 'regression':
            csv_dict = {'header': ['gname', 'pragma']}
            testr, loss_dict, encode_loss, eval_df = test(loader, 'test', model, 0, plot_test=True, csv_dict=csv_dict,
                                                          data_dict=data_dict, forward_pairwise=forward_pairwise,
                                                          eval_pairwise=eval_pairwise, test_name=test_name)
            assert eval_df.iloc[-1]['target'] == 'tot/avg'
            rmse = eval_df.iloc[-1]['rmse']
            final_print_dict[test_name] = f'{rmse:.4f}'
            saver.log_info(f'{loss_dict}')
            saver.log_info(f'{test_name} loss: {testr:.7f}, encode loss: {encode_loss:.7f}')
            saver.log_dict_of_dicts_to_csv(f'{test_name} actual-prediction', csv_dict, csv_dict['header'])
        else:
            testr, loss_dict, encode_loss, eval_df = test(loader, 'test', model, 0, forward_pairwise=False, eval_pairwise=False)

            accuracy = eval_df['accuracy']
            final_print_dict[test_name] = f'{accuracy:.4f}'

            saver.log_info(('Test loss: {:.3f}'.format(testr)))
        saver.log_info(f'-' * 100)
        saver.log_info('')
    saver.log_info(f'FLAGS.load_model=\n{FLAGS.load_model}\n{basename(dirname(FLAGS.load_model))}\n{loaded_model_info}')

    ks = "\t".join(final_print_dict.keys())
    vs = "\t".join(final_print_dict.values())
    saver.log_info(f'Summary of inference:')
    saver.log_info(f'{ks}')
    saver.log_info(f'{vs}')

    if FLAGS.save_emb:
        saver.save_emb_save_to_disk(f'embeddings.pickle')

    saver.log_info(f'Inference done')

    # _gather_eval_pair_data(dataset)


def test(loader, tvt, model, epoch, plot_test=False, test_losses=[-1], csv_dict=None,
         data_dict=None, forward_pairwise=False, eval_pairwise=False, test_name=None):
    global pred
    model.eval()

    inference_loss = 0
    correct, total = 0, 0
    # i = 0

    loss_dict, target_list = create_loss_dict_get_target_list(FLAGS)

    points_dict = create_pred_dict(target_list)
    points_pred_by_gname = defaultdict(OrderedDict)
    pairs_pred_by_gname = defaultdict(OrderedDict)
    for test_iter, data in enumerate(tqdm(loader)):
        data = data.to(FLAGS.device)
        with torch.no_grad():  # TODO: double check this to ensure no problem
            out_dict, loss, loss_dict_, gae_loss = model(data, forward_pairwise=forward_pairwise, tvt=tvt,
                                                         iter=test_iter, test_name=test_name)
        # if not FLAGS.multi_target:
        #     out_dict = {}
        #     out_dict[FLAGS.target] = out
        # else:
        #     out_dict = out
        # pred = out.round().to(torch.long)
        if FLAGS.task == 'regression':
            total += loss.item()
            if not FLAGS.SSL:
                loss_dict = update_loss_dict(loss_dict, loss_dict_, target_list, FLAGS)
        else:
            loss, pred = torch.max(out_dict[FLAGS.target[0]], 1)
            labels = _get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        if not FLAGS.SSL:
            for target_name in target_list:
                # if FLAGS.subtask == 'inference':
                #     saver.info(f'{target_name}')
                out = _get_out_from_out_dict(out_dict, target_name)
                for i in range(len(out)):
                    out_value = out[i].item()
                    if FLAGS.encode_log and target_name == 'actual_perf':
                        out_value = 2 ** (out_value) * (1 / FLAGS.normalizer)
                    # if FLAGS.subtask == 'inference':
                        # inference_loss += inference_loss_function(out_value,
                        #                                           _get_y_with_target(data, target_name)[i].item())

                        # if csv_dict is not None and not forward_pairwise and not FLAGS.sequence_modeling:
                        #     gname = _get_y_with_target(data, 'gname')[i]
                        #     # pragma = _get_y_with_target(data, 'pragmas')[i][0].item()
                        #     all_pragma = ''
                        #     for j in _get_y_with_target(data, 'pragmas')[i]:
                        #         all_pragma += (str(j.item()) + '-')
                        #     pragma = all_pragma
                        #     if f'{gname}-{pragma}' not in csv_dict:
                        #         csv_dict[f'{gname}-{pragma}'] = {'gname': gname, 'pragma': pragma}
                        #     csv_dict[f'{gname}-{pragma}'][f'acutal-{target_name}'] = \
                        #         _get_y_with_target(data, target_name)[i].item()
                        #     csv_dict[f'{gname}-{pragma}'][f'predicted-{target_name}'] = out_value
                        #     l = csv_dict['header']
                        #     if f'acutal-{target_name}' not in l:
                        #         l.extend([f'acutal-{target_name}', f'predicted-{target_name}'])
                        #         csv_dict['header'] = l

                        # if out_value != _get_y_with_target(data, target_name)[i].item():
                        #     saver.info(
                        #         f"data {i} {_get_y_with_target(data, 'gname')[i]} pramga {_get_y_with_target(data, 'pragmas')[i][0].item()} actual value: {_get_y_with_target(data, target_name)[i].item():.2f}, predicted value: {out_value:.2f}")
                    true_val = _get_y_with_target(data, target_name)[i].item()
                    points_dict[target_name]['pred'].append(out_value)
                    points_dict[target_name]['true'].append(true_val)

                if FLAGS.subtask == 'inference' and eval_pairwise:
                    if forward_pairwise:
                        li = data.xy_dict_programl['point']
                        assert type(li) is list and len(li) == 1
                        points_li = li[0]
                        assert len(points_li) % 2 == 0
                        sp = len(points_li) // 2
                        for i in range(sp):
                            key1, key2 = points_li[i], points_li[sp + i]
                            d = {target_name: torch.argmax(
                                _get_out_from_out_dict(out_dict, f'{target_name}_pairwise_class')[i]).item()
                                 for target_name in target_list}
                            # d['emb_T'] = out_dict['emb_T'][i].detach().cpu().numpy() TODO
                            pairs_pred_by_gname[data.gname[0][i]][(key1, key2)] = d
                    else:
                        if not FLAGS.sequence_modeling:  # TODO: enable it
                            for i, data_key in enumerate(data.xy_dict_programl['point']):
                                d = {target_name: _get_out_from_out_dict(out_dict, target_name)[i].item()
                                     for target_name in target_list}
                                d['emb_T'] = out_dict['emb_T'][i].detach().cpu().numpy()
                                points_pred_by_gname[data.gname[i]][data_key] = d

        # corrects.append(pred.eq(data.y.to(torch.long)))
        # total_ratio += ratio
        # if i % FLAGS.print_every_iter == 0:
        #     print(f'Iter {i + 1}: Loss {loss}')
        # i += 1

        # break  # TODO: just for debugging
    if FLAGS.plot_pred_points and tvt == 'test' and (
            plot_test or (test_losses and (total / len(loader)) < min(test_losses))):
        from utils import plot_points, plot_points_with_subplot
        saver.log_info(f'@@@ plot_pred_points {test_name}')
        if not FLAGS.multi_target:
            plot_points({f'{FLAGS.target[0]}-pred_points': points_dict[f'{FLAGS.target[0]}']['pred'],
                         f'{FLAGS.target[0]}-true_points': points_dict[f'{FLAGS.target[0]}']['true']},
                        f'epoch_{epoch + 1}_{tvt}_{test_name}', saver.plotdir)
            print(f'done plotting with {correct} corrects out of {total}')
        else:
            assert (isinstance(FLAGS.target, list))
            plot_points_with_subplot(points_dict,
                                     f'epoch_{epoch + 1}_{tvt}_{test_name}', saver.plotdir, target_list)

    eval_df = {}
    if FLAGS.subtask in ['inference', 'train']:
        if FLAGS.task == 'regression':
            eval_df = _report_rmse_etc(points_dict, f'epoch {epoch}:', True)
            if eval_pairwise:
                saver.log_info(f'len(points_pred_by_gname)={len(points_pred_by_gname)}')
                tot_points = 0
                for gname, d in points_pred_by_gname.items():
                    saver.log_info(f'\t{gname}: {len(d)} points', silent=True)
                    tot_points += len(d)
                saver.log_info(f'{tot_points} points in total')
                saver.log_info(f'len(pairs_pred_by_gname)={len(pairs_pred_by_gname)}')
                for gname, d in pairs_pred_by_gname.items():
                    saver.log_info(f'\t{gname}: {len(d)} pairs')
                pair_dict, pred_dict_by_target_global = gather_eval_pair_data(data_dict, points_pred_by_gname,
                                                                              pairs_pred_by_gname,
                                                                              target_list, test_name)
                if points_pred_by_gname is not None:
                    _report_class_result(pred_dict_by_target_global,
                                         f'{len(data_dict)}_pairwise_kernels_pred_dict_by_target_global:')
        elif FLAGS.task == 'class':
            eval_df = report_class_loss(points_dict)
        else:
            raise NotImplementedError()

    if FLAGS.task == 'regression':
        if FLAGS.subtask == 'inference':
            rtn = (total / len(loader), {key: f'{(v / len(loader)):.4f}' for key, v in loss_dict.items()},
                   inference_loss / len(loader) / FLAGS.batch_size, eval_df)
        else:
            rtn = total / len(loader), {key: f'{(v / len(loader)):.4f}' for key, v in
                                        loss_dict.items()}, gae_loss.item(), eval_df
    else:
        rtn = 1 - correct / total, {key: f'{(v / len(loader)):.4f}' for key, v in
                                    loss_dict.items()}, gae_loss.item(), eval_df
    # saver.log_info(f'len(loader)={len(loader)}')
    return rtn


def _report_class_result(points_dict, label):
    saver.log_info(label)
    for target_name, d in points_dict.items():
        labels = d['true']
        pred = d['pred']
        target_names = ['d1 <= d2', 'd1 > d2']
        assert len(labels) == len(pred)
        saver.log_info(f'-----\n{target_name} classification report {label} ({len(labels)} data points)')
        saver.log_info(classification_report(labels, pred, target_names=target_names, digits=4))
        cm = confusion_matrix(labels, pred, labels=[0, 1])
        saver.log_info(f'Confusion matrix:\n{cm}')
        emb_diff_li = d.get('emb_diff')
        # print('@@@@@'*100)
        # print('emb_diff_li is', emb_diff_li)
        if emb_diff_li is not None:
            assert type(emb_diff_li) is list
            print_stats(emb_diff_li, 'emb_diff_li', saver=saver)
        saver.log_info(f'\n----')


def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = d['true']
    pred = d['pred']
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    s = classification_report(labels, pred, target_names=target_names, digits=4)
    saver.log_info(s)
    # cm = confusion_matrix(labels, pred, labels=[0, 1])
    # saver.info(f'Confusion matrix:\n{cm}')
    return classification_report(labels, pred, target_names=target_names, output_dict=True)


def _report_rmse_etc(points_dict, label, print_result=True):
    if print_result:
        saver.log_info(label)
    data = defaultdict(list)
    tot_mape, tot_rmse, tot_mse, tot_mae, tot_max_err, tot_tau, tot_std = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    num_data = None
    try:
        for target_name, d in points_dict.items():
            # true_li = d['true']
            # pred_li = d['pred']
            true_li = d['true']
            pred_li = d['pred']
            num_data = len(true_li)
            mape = mean_absolute_percentage_error(true_li, pred_li)
            rmse = mean_squared_error(true_li, pred_li, squared=False)
            mse = mean_squared_error(true_li, pred_li, squared=True)
            mae = mean_absolute_error(true_li, pred_li)
            max_err = max_error(true_li, pred_li)

            true_rank = rankdata(true_li)
            pred_rank = rankdata(pred_li)
            tau = kendalltau(true_rank, pred_rank)[0]
            data['target'].append(target_name)
            data['mape'].append(mape)
            data['rmse'].append(rmse)
            data['mse'].append(mse)
            data['mae'].append(mae)
            data['max_err'].append(max_err)
            data['tau'].append(tau)

            # data['rmse'].append(f'{rmse:.4f}')
            # data['mse'].append(f'{mse:.4f}')
            # data['tau'].append(f'{tau: .4f}')
            tot_mape += mape
            tot_rmse += rmse
            tot_mse += mse
            tot_mae += mae
            tot_max_err += max_err
            tot_tau += tau

            pred_std = d.get('pred_std')
            if pred_std is not None:
                assert type(pred_std) is np.ndarray, f'{type(pred_std)}'
                pred_std = np.mean(pred_std)
                data['pred_std'].append(pred_std)
                tot_std += pred_std
        data['target'].append('tot/avg')
        data['mape'].append(tot_mape)
        data['rmse'].append(tot_rmse)
        data['mse'].append(tot_mse)
        data['mae'].append(tot_mae)
        data['max_err'].append(tot_max_err)
        data['tau'].append(tot_tau / len(points_dict))
        if 'pred_std' in data:
            data['pred_std'].append(tot_std / len(points_dict))
    except ValueError as v:
        saver.log_info(f'Error {v}')
        data = defaultdict(list)

    # data['rmse'].append(f'{tot_rmse:.4f}')
    # data['mse'].append(f'{tot_mse:.4f}')
    # data['tau'].append(f'{tot_tau / len(points_dict):.4f}')
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    if print_result:
        saver.log_info(num_data)
        saver.log_info(df.round(4))
    # exit()
    return df
    # exit()


def _get_out_from_out_dict(out_dict, target_name):
    if FLAGS.task == 'class':
        out = pred
    elif FLAGS.encode_log and 'perf' in target_name:
        out = out_dict['perf']
    else:
        out = out_dict[target_name]
    return out

# def inference_loss_function(pred, true):
#     return (pred - true) ** 2
