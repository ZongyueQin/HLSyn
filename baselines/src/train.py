from test import test
from train_pairwise import get_pairwise_data_loaders

from config import FLAGS
from saver import saver
from utils import OurTimer, _get_y_with_target, create_loss_dict_get_target_list, update_loss_dict
from data import get_kernel_samples, split_dataset, torch_generator

from model import feature_extract, check_feature_extract
from model_factory import create_model

import torch
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join, basename


def train_main(dataset, pragma_dim=None):
    # saver.info(f'Reading dataset from {SAVE_DIR}')

    # from torch.utils.data import random_split  # TODO: inductive

    if not FLAGS.all_kernels:
        dataset = get_kernel_samples(dataset)

    train_loaders = []
    if FLAGS.pairwise_class:
        # if FLAGS.force_regen_pairwise_data:
        #     pair_dict = _gather_eval_pair_data(dataset)
        # else:
        #     pair_dict = None
        train_loader, val_loader, test_loader = get_pairwise_data_loaders(dataset, torch_generator)
        # for data in train_loader:
        #     print(data)
        # exit()
        train_loaders.append((train_loader, 'p'))
        if FLAGS.both_train_loaders:
            train_loader_regular, *_ = split_dataset(dataset, torch_generator)
            train_loaders.append((train_loader_regular, 'r'))
    else:
        train_loader, val_loader, test_loader, _ = split_dataset(dataset, torch_generator)
        train_loaders.append((train_loader, 'r'))

    model = create_model(train_loader, pragma_dim)
    saver.log_model_architecture(model)

    if FLAGS.sequence_modeling and not FLAGS.finetune_bert:
        for name, param in model.bert_model.named_parameters():
            param.requires_grad = False
            saver.log_info(f'No fine tune bert: Freezing param {name}')

    if FLAGS.load_model != None and FLAGS.load_model != 'None':
        saver.load_trained_model(FLAGS.load_model, model)

        if FLAGS.pairwise_class and FLAGS.loss_components == 'both' and FLAGS.fix_encoder_classMLPs:
            saver.log_info(f'fix_encoder_classMLPs=True:')
            except_params = set([f'MLPs.{x[0]}' for x in list(model.MLPs.named_parameters())])
            for name, param in model.named_parameters():
                if name not in except_params:
                    saver.log_info(f'\tFixing parameter: {name}')
                    param.requires_grad = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    saver.log_info(f'\tAllow training: {name}')
    else:
        saver.info(f'FLAGS.load_model is None')

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
    # if not FLAGS.multi_target:
    #     model = Net(num_features).to(FLAGS.device)
    # else:
    #     model = NetMultiObjective(num_features).to(FLAGS.device)
    # print(model)

    optimizer = create_optimizer(model)

    # Initialize to optimal attention weights:
    # model.pool1.weight.data = torch.tensor([0., 1., 0., 0.]).view(1,4).to(device)
    train_losses = []
    val_losses = []
    test_losses = []
    gae_train_losses = []
    gae_val_losses = []
    gae_test_losses = []
    epochs = range(FLAGS.epoch_num)
    plot_test = False
    for epoch in epochs:
        print(basename(saver.logdir))
        plot_test = False
        timer = OurTimer()
        if FLAGS.feature_extract:
            check_feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
        saver.log_info(f'Epoch {epoch} starts')

        # val
        if FLAGS.ignore_validation:
            saver.log_info('Ignore validation')
            val = 0.0
            loss_dict_val = {}
            gae_loss_val = 0.0
        else:
            if len(val_loader) > 0:
                saver.log_info(f'Epoch {epoch} val')
                val, loss_dict_val, gae_loss_val, _ = test(val_loader, 'val', model, epoch,
                                                           forward_pairwise=FLAGS.pairwise_class, eval_pairwise=False)
                saver.writer.add_scalar('val/val', val, epoch)

                if FLAGS.save_model and epoch % FLAGS.save_every_epoch == 0:
                    saver.save_trained_model(model, f"_{epoch}")

                if val_losses and val < min(val_losses):
                    if FLAGS.save_model:
                        saver.log_info((f'Saved val model at epoch {epoch}'))
                        # torch.save(model.state_dict(), join(saver.logdir, "val_model_state_dict.pth"))
                        saver.save_trained_model(model, path=join(saver.logdir, "val_model_state_dict.pth"),
                                                 info={'epoch': epoch})
                        # saver.save_trained_model(model, f"_val_{epoch}")
                    plot_test = True

        # test
        if FLAGS.ignore_testing:
            saver.log_info('Ignore testing')
            testr = 0.0
            loss_dict_test = {}
            gae_loss_test = 0.0
        else:
            if len(test_loader) > 0:
                saver.log_info(f'Epoch {epoch} test')
                testr, loss_dict_test, gae_loss_test, _ = test(test_loader, 'test', model, epoch, plot_test,
                                                               test_losses,
                                                               forward_pairwise=False,
                                                               eval_pairwise=False)
                saver.writer.add_scalar('test/test', testr, epoch)

                if test_losses and testr < min(test_losses):
                    if FLAGS.save_model:
                        saver.log_info((f'Saved test model at epoch {epoch}'))
                        # torch.save(model.state_dict(), join(saver.logdir, "test_model_state_dict.pth"))
                        saver.save_trained_model(model, path=join(saver.logdir, "test_model_state_dict.pth"),
                                                 info={'epoch': epoch})

        if len(val_loader) > 0 and len(test_loader) > 0:
            saver.log_info((f'Val GAE loss: {gae_loss_val}'))
            saver.log_info((f'Val loss breakdown {loss_dict_val}'))
            saver.log_info((f'Test GAE loss: {gae_loss_test}'))
            saver.log_info((f'Test loss breakdown {loss_dict_test}'))
            val_losses.append(val)
            test_losses.append(testr)
            gae_val_losses.append(gae_loss_val)
            gae_test_losses.append(gae_loss_test)
        elif len(test_loader) > 0:
            saver.log_info((f'Test GAE loss: {gae_loss_test}'))
            saver.log_info((f'Test loss breakdown {loss_dict_test}'))
            test_losses.append(testr)
            gae_test_losses.append(gae_loss_test)

        # train
        saver.log_info(f'Epoch {epoch} train')
        train_loader_chosen, train_loader_name = train_loaders[epoch % len(train_loaders)]
        forward_pairwise = FLAGS.pairwise_class
        if len(train_loaders) > 1:
            saver.log_info(f'Epoch {epoch}: Choose {train_loader_name}')
            saver.writer.add_text('train/train_loader_chosen', train_loader_name, epoch)
            if train_loader_name == 'r':
                forward_pairwise = False
        loss, loss_dict_train, gae_loss_train, _ = train(epoch, model, train_loader_chosen, optimizer, forward_pairwise)

        if train_losses and loss < min(train_losses):
            if FLAGS.save_model:
                saver.log_info((f'Saved train model at epoch {epoch}'))
                # torch.save(model.state_dict(), join(saver.logdir, "train_model_state_dict.pth"))
                saver.save_trained_model(model, path=join(saver.logdir, "train_model_state_dict.pth"),
                                         info={'epoch': epoch})
            # plot_test = True

        saver.log_info((f'Train GAE loss: {gae_loss_train}'))
        saver.log_info((f'Train loss breakdown {loss_dict_train}'))

        if len(val_loader) > 0 and len(test_loader) > 0:
            saver.log_info(('Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, '
                            'Test: {:.4f}) Time: {}'.format(
                epoch, loss, val, testr, timer.time_and_clear())))
        elif len(test_loader) > 0:
            saver.log_info(('Epoch: {:03d}, Train loss: {:.4f}, '
                            'Test: {:.4f}) Time: {}'.format(
                epoch, loss, testr, timer.time_and_clear())))
        else:
            saver.log_info(('Epoch: {:03d}, Train loss: {:.4f}, '
                            'Time: {}'.format(
                epoch, loss, timer.time_and_clear())))

        train_losses.append(loss)
        saver.writer.add_scalar('loss/loss_epoch', loss, epoch)
        gae_train_losses.append(gae_loss_train)

        if len(train_losses) > 50:
            if len(set(train_losses[-50:])) == 1 and len(set(test_losses[-50:])) == 1:
                break

    if FLAGS.epoch_num == 0:
        return

    epochs = range(epoch + 1)
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    if len(val_loader) > 0:
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
    if len(test_loader) > 0:
        plt.plot(epochs, test_losses, 'r', label='Testing loss')
    plt.title('Training, Validation, and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(saver.get_log_dir(), 'losses.png'), bbox_inches='tight')
    plt.close()

    if FLAGS.gae_T or FLAGS.gae_P:
        plt.plot(epochs, gae_train_losses, 'g', label='Training loss')
        if len(val_loader) > 0:
            plt.plot(epochs, gae_val_losses, 'b', label='Validation loss')
        if len(test_loader) > 0:
            plt.plot(epochs, gae_test_losses, 'r', label='Testing loss')
        plt.title('Training, Validation, and Testing loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(join(saver.get_log_dir(), 'gae_losses.png'), bbox_inches='tight')
        plt.close()
    if len(test_loader) > 0:
        saver.log_info(f'min test loss at epoch: {test_losses.index(min(test_losses))}')
    if len(val_loader) > 0:
        saver.log_info(f'min val loss at epoch: {val_losses.index(min(val_losses))}')
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses))}')


def create_optimizer(model):
    if FLAGS.opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    elif FLAGS.opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    else:
        raise ValueError(f'FLAGS.opt_type {FLAGS.opt_type} unrecognized')
    return optimizer


def train(epoch, model, train_loader, optimizer, forward_pairwise):
    model.train()

    total_loss = 0
    correct = 0
    i = 0
    num_data = 0

    loss_dict, target_list = create_loss_dict_get_target_list(FLAGS)

    for iter_id, data in enumerate(tqdm(train_loader)):
        num_data += data.num_graphs
        if FLAGS.debug_iter != -1 and iter_id > FLAGS.debug_iter:
            saver.log_info(f'Debugging mode: iter_id={iter_id} > FLAGS.debug_iter={FLAGS.debug_iter}; stop the epoch')
            break

        # data = data.to(FLAGS.device)
        optimizer.zero_grad()
        out, loss, loss_dict_, gae_loss = model(data.to(FLAGS.device), forward_pairwise=forward_pairwise, tvt='train')

        if FLAGS.load_pretrained_GNN:
            # outs = model.pretrained_GNN_encoder.get_node_emb(data)

            if FLAGS.model != 'our' or (FLAGS.sequence_modeling and not FLAGS.multi_modality):
                pass  # no GNN
            else:
                node_emb = out['node_emb']
                x = data.x
                if x is None:
                    x = data.x_programl
                node_emb = node_emb[x[:, 3] == 0]  # non-pragma nodes
                # node_emb, _ = model.get_node_emb(data)
                # node_emb = node_emb[data.x[:, 3] == 0]
                guide_node_emb, _ = model.pretrained_GNN_encoder.get_node_emb(data)
                guide_node_emb = guide_node_emb[x[:, 3] == 0]  # non-pragma nodes
                assert node_emb.shape == guide_node_emb.shape, f'node_emb.shape={node_emb.shape}; guide_node_emb.shape={guide_node_emb.shape}'
                guide_loss = torch.mean(1 - F.cosine_similarity(node_emb, guide_node_emb.detach()))
                # print(guide_loss)
                loss += guide_loss
                loss_dict_['guide_loss'] = guide_loss

        # out, loss, loss_dict_, gae_loss = model(data)
        # loss = ((out - data.y).pow(2) + 100 * attn_loss).mean()
        loss.backward()
        if hasattr(FLAGS, 'max_grad_norm') and FLAGS.max_grad_norm is not None and FLAGS.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_grad_norm)
            saver.log_info_at_most(f'torch.nn.utils.clip_grad_norm_ FLAGS.max_grad_norm={FLAGS.max_grad_norm}', msg_type='gclip', times=1)

        if FLAGS.task == 'regression':
            total_loss += loss.item() * data.num_graphs
            if not FLAGS.SSL:
                loss_dict = update_loss_dict(loss_dict, loss_dict_, target_list, FLAGS)
        else:
            loss_, pred = torch.max(out[FLAGS.target[0]], 1)
            labels = _get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total_loss += labels.size(0)
        optimizer.step()
        saver.writer.add_scalar('loss/loss', loss.item(), epoch * len(train_loader) + i)
        for loss_name, loss_v in loss_dict_.items():
            saver.writer.add_scalar(f'train/{loss_name}', loss_v.item(), epoch * len(train_loader) + i)
        # if i % FLAGS.print_every_iter == 0:
        #     print(f'Iter {i + 1}: Loss {loss}')
        i += 1

    if FLAGS.task == 'regression':
        return total_loss / len(train_loader), {key: f'{(v / len(train_loader)):.4f}' for key, v in
                                                loss_dict.items()}, gae_loss.item(), num_data
    else:
        return 1 - correct / total_loss, {key: f'{(v / len(train_loader)):.4f}' for key, v in
                                          loss_dict.items()}, gae_loss.item(), num_data
