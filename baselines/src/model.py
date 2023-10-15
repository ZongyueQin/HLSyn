from graph_transfotmer import get_num_features_graph_transformer, GPSConv
from data_src_code import create_code_encoder
from train_pairwise import get_comp_result_tagret
from data import get_num_features
from nn import MyGlobalAttention
from config import FLAGS
from saver import saver
from utils_nn import create_graph_att_module
from utils import MLP, create_act, estimate_model_size
import data

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, TransformerConv, GCNConv, GINEConv
from torch_geometric.nn import global_add_pool
import torch.nn as nn

from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU

from collections import OrderedDict


class Model(nn.Module):
    def _create_src_code_bert(self):
        saver.log_info('Creating bert_model')
        bert_model, BERT_DIM = create_code_encoder()
        saver.log_info('Created bert_model')
        estimate_model_size(bert_model, f'bert_model {FLAGS.code_encoder}', saver)
        return bert_model, BERT_DIM

    def _get_GNN_conv_class(self):
        if FLAGS.graph_transformer_option is not None:
            conv_class = None
        else:
            if FLAGS.gnn_type == 'gat':
                conv_class = GATConv
            elif FLAGS.gnn_type == 'gcn':
                conv_class = GCNConv
            elif FLAGS.gnn_type == 'transformer':
                conv_class = TransformerConv
            else:
                raise NotImplementedError()
        return conv_class

    def _create_conv_first(self, edge_dim, in_channels, D, conv_class, sequence_modeling, hidden_channels=None):
        if sequence_modeling:
            if hidden_channels is None:
                hidden_channels = [self.bert_dim // 2, self.bert_dim // 4]
            conv_first = MLP(in_channels, D, activation_type=FLAGS.activation,
                             hidden_channels=hidden_channels, num_hidden_lyr=2)
        else:
            if FLAGS.graph_transformer_option is not None:
                gt_dim = get_num_features_graph_transformer(in_channels)
                conv_first = MLP(gt_dim, D, activation_type=FLAGS.activation, num_hidden_lyr=2)
            else:
                if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                    # print(in_channels)
                    conv_first = conv_class(in_channels, D, edge_dim=edge_dim)
                else:
                    conv_first = conv_class(in_channels, D)
        return conv_first

    def _create_GNN_conv_layers(self, num_layers, edge_dim, D, conv_class):
        conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            if FLAGS.graph_transformer_option is not None:
                gt_conv_type = FLAGS.graph_transformer_option.get('conv_type', 'mha')
                if gt_conv_type == 'mha':
                    conv = nn.MultiheadAttention(D, num_heads=FLAGS.graph_transformer_option['num_heads'],
                                                 batch_first=True)
                elif gt_conv_type == 'gps_conv':
                    if FLAGS.graph_transformer_option['need_local_mpnn']:
                        local_mpnn = GINEConv(Sequential(Linear(D, D), create_act(FLAGS.activation), Linear(D, D)),
                                              edge_dim=edge_dim)
                    else:
                        local_mpnn = None
                    conv = GPSConv(D, conv=local_mpnn, heads=FLAGS.graph_transformer_option['num_heads'])
                else:
                    raise NotImplementedError()
            else:
                if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                    conv = conv_class(D, D, edge_dim=edge_dim)
                else:
                    conv = conv_class(D, D)
            conv_layers.append(conv)
        jkn = JumpingKnowledge(FLAGS.jkn_mode, channels=D, num_layers=2)
        return conv_layers, jkn

    def _get_target_list(self):
        if 'regression' in self.task:
            _target_list = FLAGS.target
            if not isinstance(FLAGS.target, list):
                _target_list = [FLAGS.task]
            # if FLAGS.new_speedup == False:
            #     self.target_list = [t for t in _target_list if t != 'perf' else 'kernel_speedup'] # to use with trained model from old speedup
            # else
            target_list = [t for t in _target_list]
        else:
            target_list = ['perf']
        return target_list

    def _create_decoder_MLPs(self, input_dim, D, target_list, out_dim, hidden_channels=None):
        MLPs = None
        MLPs_pairwise_class = None
        if FLAGS.node_attention:
            if FLAGS.separate_P_T:
                in_D = 2 * input_dim
            else:
                in_D = input_dim
        else:
            in_D = input_dim
        if hidden_channels is None:
            if D > 64:
                hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
            else:
                hidden_channels = [D // 2, D // 4, D // 8]
        if not FLAGS.SSL:
            MLPs = self._create_target_MLPs_dict(target_list, hidden_channels, in_D, out_dim)
            if FLAGS.pairwise_class:
                in_D_class = in_D * len(FLAGS.comp_ops)
                # TODO: choice: compare before each target or for/within/dedicated to each target
                MLPs_pairwise_class = self._create_target_MLPs_dict(target_list, hidden_channels, in_D_class,
                                                                    2)  # binary
        return MLPs, MLPs_pairwise_class

    def _create_target_MLPs_dict(self, target_list, hidden_channels, in_D, out_D):
        rtn = nn.ModuleDict()
        for target in target_list:
            rtn[target] = MLP(in_D, out_D, activation_type=FLAGS.activation,
                              hidden_channels=hidden_channels,
                              num_hidden_lyr=len(hidden_channels))
        return rtn

    def _create_loss(self):
        if FLAGS.task == 'regression':
            out_dim = 1
            loss_function = nn.MSELoss()
        else:
            out_dim = 2
            loss_function = nn.CrossEntropyLoss()
        return out_dim, loss_function

    def _apply_bert(self, x, data, additional_x=None, nti_objs=None):
        x_shape = x.shape
        attention_mask = None
        if FLAGS.token_att_masking:
            attention_mask = data.attention_mask
        if hasattr(FLAGS, 'vis_transformer_att') and FLAGS.vis_transformer_att:
            output_attentions = True
        else:
            output_attentions = False
        if FLAGS.code_encoder == 'codet5':
            x_bert = self.bert_model(x, attention_mask=attention_mask, output_attentions=output_attentions,
                                     additional_x=additional_x, nti_objs=nti_objs)
        else:
            x_bert = self.bert_model(x, attention_mask=attention_mask, output_attentions=output_attentions)
        if hasattr(FLAGS, 'vis_transformer_att') and FLAGS.vis_transformer_att:
            from data_src_code import tokenizer
            from bertviz import head_view, model_view
            attentions = x_bert['attentions']
            # Just vis the 0th sequence.
            attention = tuple(
                torch.unsqueeze(a[0], dim=0) for a in attentions)  # taking the 0th seq/chunk's attention maps
            tokens = tokenizer.convert_ids_to_tokens(x[0])
            saver.log_info(f'tokens={tokens}')
            for layer_id, a in enumerate(attention):
                assert a.shape[0] == 1
                num_heads = a.shape[1]
                for head_id in range(num_heads):
                    vec = a[0][head_id][0]
                    top_10_ids = torch.argsort(vec, descending=True)[0:10]
                    top_10_tokens = [tokens[token_id] for token_id in top_10_ids]
                    saver.log_info(
                        f'Token {tokens[0]}\'s top 10 attention tokens layer {layer_id} head {head_id}: {top_10_tokens}')
            f = ' '.join(tokens)
            saver.log_info(f'full string={f}')
            html_head_view = head_view(attention, tokens, html_action='return')
            with open(f"{saver.get_obj_dir()}/head_view.html", 'w') as file:
                file.write(html_head_view.data)
            saver.log_info(f'vis_transformer_att done; exiting')
            exit()
        if FLAGS.chunk_emb == 'pooler':
            x_bert_rtn = x_bert['pooler_output']
        elif FLAGS.chunk_emb == 'cls':
            # from src_code_modeling import tokenizer
            # xxx = tokenizer("Hello, my dog is cute", return_tensors="pt")
            # yyy = self.bert_model(**xxx)
            # saver.log_info(yyy)
            #
            # saver.log_info(x_bert)
            # saver.log_info(f'last_hidden_state')
            # saver.log_info(x_bert.last_hidden_state)
            # saver.log_info(x_bert.last_hidden_state.shape)
            # exit()
            x_bert_rtn = x_bert['last_hidden_state'][:, 0, :]
        else:
            assert False
        assert x_bert_rtn.shape == (x_shape[0], self.bert_dim)
        return x_bert_rtn, x_bert

    def _get_act_func(self):
        if FLAGS.activation == 'relu':
            activation = F.relu
        elif FLAGS.activation == 'elu':
            activation = F.elu
        else:
            raise NotImplementedError()
        return activation

    def _apply_conv_first(self, conv_first, x, edge_index, edge_attr, activation):
        if FLAGS.graph_transformer_option is not None:
            out = conv_first(x)
        else:
            if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                out = conv_first(x, edge_index, edge_attr=edge_attr)
            else:
                # try:
                if len(edge_index) == 0:
                    raise ValueError(f'len(edge_index) == {len(edge_index)}; skip GNN')
                else:
                    # if FLAGS.sequence_modeling:
                    #     out = x
                    # else:
                    out = conv_first(x, edge_index)
                # except Exception as e:
                #     print(e)
        if activation is not None:
            out = activation(
                out)  # this is tricky! MLP does not apply activation for the last layer, but here we may apply it at the end
        return out

    def _apply_conv_layers(self, out, conv_layers, jkn, outs, edge_index, edge_attr, batch, data, activation):
        for i, conv in enumerate(conv_layers):
            if FLAGS.graph_transformer_option is not None:
                gt_conv_type = FLAGS.graph_transformer_option.get('conv_type', 'mha')
                if gt_conv_type == 'mha':
                    attention_map_aug = FLAGS.graph_transformer_option.get('attention_map_aug')
                    if attention_map_aug is not None:
                        attn_mask = data.att_mask_tensor
                    else:
                        attn_mask = None
                    out, _ = conv(query=out, key=out, value=out, attn_mask=attn_mask)
                elif gt_conv_type == 'gps_conv':
                    out = conv(out, edge_index, batch, edge_attr=edge_attr)
                else:
                    raise NotImplementedError()
            else:
                if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                    if edge_attr is not None:
                        assert edge_attr.max() == 1, f'edge_attr wrong! Should be binary; max() is {edge_attr.max()}'  # to ensure binary vectors
                    out = conv(out, edge_index, edge_attr=edge_attr)
                else:
                    out = conv(out, edge_index)
            if i != len(conv_layers) - 1:
                out = activation(out)

            if FLAGS.ptrans and FLAGS.early_late_fusion == 'early':
                for tstype in self.seen_pragma_trans_types:
                    if FLAGS.MLPs_share_weights:
                        ts_mlp = self.tstype_MLPs[tstype]
                    else:
                        ts_mlp = self.tstype_MLPs[f'{tstype}_{i}']
                    out = self._ptrans_apply_MLP(out, ts_mlp, data, tstype)

            outs.append(out)

        if FLAGS.jkn_enable:
            out = jkn(outs)
        return out, outs

    def _gen_graph_level_embedding_from_node(self, node_emb, batch, data, out_dict, glob_P, glob_T, glob,
                                             att_mask=None):
        out_P = None
        if FLAGS.node_attention:
            if FLAGS.separate_P_T:
                # print(data.gname)'
                if FLAGS.P_use_all_nodes:
                    out_P, node_att_scores_P = glob_P(node_emb, batch, att_mask=att_mask)
                else:
                    out_P, node_att_scores_P = glob_P(node_emb, batch, set_zeros_ids=data.X_contextnids,
                                                      att_mask=att_mask)

                out_T, node_att_scores = glob_T(node_emb, batch, set_zeros_ids=data.X_pragmanids, att_mask=att_mask)
                '''
                All of the following do not work due to the inplace modification not allowed 
                (would lead to RuntimeError).
                # node_emb[torch.nonzero(data.X_pragmanids)] = node_emb[torch.nonzero(data.X_pragmanids)].clone() + 5
                # node_emb[2] = node_emb[2] * node_emb[3].clone()
                # node_emb.index_add_(0, torch.tensor([0, 2, 3]), torch.ones(3, 64))
                # node_emb += 1         
                Therefore, to apply MLP/transformation to only certain nodes/rows, need to apply the MLP to all rows
                and then apply some masking.
                '''

                out_dict['emb_P'] = out_P

                out_embed = torch.cat((out_P, out_T), dim=1)

            else:
                out_T, node_att_scores = glob(node_emb, batch, att_mask=att_mask)
                # if FLAGS.subtask == 'visualize' and FLAGS.vis_what == 'att':
                #     saver.save_dict({'data': data, 'node_att_scores': node_att_scores},
                #                     f'{tvt}_{epoch}_{iter}_node_att.pickle')
                out_embed = out_T
        else:
            out_T = global_add_pool(node_emb, batch)
            out_embed = out_T
        out_dict['emb_T'] = out_T
        return out_dict, out_embed, out_P, out_T

    def _apply_target_MLPs_with_loss(self, mlps, out_embed, data, total_loss, out_dict, loss_dict, mode):
        for target_name in self.target_list:
            out = mlps[target_name](out_embed)
            y = get_y_with_target(data, target_name)
            if self.task == 'regression':
                target = y.view((len(y), self.out_dim))
            else:
                target = y.view((len(y)))

            if mode == 'normal':
                # if self.task == 'regression' and FLAGS.loss_func == 'mse_weighted_util':
                #     loss = self.loss_function(out, target, target_name=target_name)
                # else:
                loss = self.loss_function(out, target)
                target_name_s = target_name
            elif mode == 'pairwise_class':
                y1, y2 = self._split_vec_mat_into_2_halves(target)
                target = get_comp_result_tagret(y1, y2)
                assert out.shape[0] == target.shape[0]
                loss = self.loss_function_pairwise_class(out, target)
                target_name_s = f'{target_name}_pairwise_class'
            else:
                assert False

            if FLAGS.loss_scale is not None:
                loss = loss * FLAGS.loss_scale[target_name]

            if FLAGS.margin_loss:
                sorted_out = out[torch.argsort(target, dim=0)].view(out.shape)
                shifted_delta = (sorted_out - torch.roll(sorted_out, -1, 0))[
                                0:-1]
                margin_loss = torch.mean(torch.max(
                    torch.zeros(shifted_delta.shape).to(FLAGS.device),
                    shifted_delta))
                print('margin loss', margin_loss)
                total_loss += margin_loss
            out_dict[target_name_s] = out

            use_regression_loss = True
            if FLAGS.pairwise_class and FLAGS.loss_components == 'class_only':
                use_regression_loss = False

            if use_regression_loss:
                total_loss += loss
            loss_dict[target_name_s] = loss
            # print(target_name, 'loss', loss)


class Net(Model):
    def __init__(self, init_pragma_dict=None, dataset=None):
        super(Net, self).__init__()

        in_channels = get_num_features(dataset)
        if FLAGS.encode_edge:
            edge_dim = dataset[0].edge_attr.shape[1]
        else:
            edge_dim = -1

        # task = FLAGS.task
        num_layers = FLAGS.num_layers
        D = FLAGS.D
        # target = FLAGS.target
        if dataset is not None:
            self.num_features_node = dataset.num_features

        if FLAGS.sequence_modeling:
            self.bert_model, self.bert_dim = self._create_src_code_bert()

        conv_class = self._get_GNN_conv_class()

        self.conv_first = self._create_conv_first(edge_dim, in_channels, D, conv_class, FLAGS.sequence_modeling)

        self.conv_layers, self.jkn = self._create_GNN_conv_layers(num_layers, edge_dim, D, conv_class)

        if FLAGS.ptrans:
            self.tstype_MLPs = self._create_ptrans_MLPs(dataset, D)

        if FLAGS.node_attention:
            if FLAGS.separate_P_T:
                self.gate_nn_T, self.glob_T = create_graph_att_module(D, return_gate_nn=True)
                if FLAGS.separate_P_T:
                    self.gate_nn_P, self.glob_P = create_graph_att_module(D, return_gate_nn=True)
                # if FLAGS.separate_pseudo:  ## for now, only pseudo node for block
                #     self.gate_nn_pseudo_B = self.node_att_gate_nn(D)
                #     self.glob_pseudo_B = MyGlobalAttention(self.gate_nn_pseudo_B, None)
            else:
                self.gate_nn, self.glob = create_graph_att_module(D, return_gate_nn=True)

        if FLAGS.gae_T:  # graph auto encoder
            if FLAGS.separate_P_T:
                self.gae_transform_T = nn.ModuleDict()
                for gname, feat_dim in init_pragma_dict.items():
                    mlp = Linear(feat_dim[0], D // 8)
                    if FLAGS.pragma_uniform_encoder:
                        self.gae_transform_T['all'] = Linear(feat_dim[1], D // 8)
                        break
                    else:
                        self.gae_transform_T[gname] = mlp
                channels = [D // 2, D // 4]
                self.decoder_T = MLP(D, D // 8,
                                     activation_type=FLAGS.activation,
                                     hidden_channels=channels,
                                     num_hidden_lyr=len(channels))
        if FLAGS.gae_P:
            out_channels = in_channels
            if FLAGS.input_encode:
                self.gate_input = Linear(in_channels, 2 * D)  ## encode input one-hot representation
                out_channels = 2 * D

            if FLAGS.decoder_type == 'type1':
                decoder_arch = []
            elif FLAGS.decoder_type == 'type2':
                decoder_arch = [D, 2 * D, out_channels]
            self.decoder_P = MLP(D, out_channels, activation_type=FLAGS.activation,
                                 hidden_channels=decoder_arch,
                                 num_hidden_lyr=len(decoder_arch))
            if FLAGS.decoder_type == 'None':
                for name, param in self.decoder_P.named_parameters():
                    print(name)
                    param.requires_grad = False
        if FLAGS.gae_T or FLAGS.gae_P:
            self.gae_sim_function = nn.CosineSimilarity()
            self.gae_loss_function = nn.CosineEmbeddingLoss()

        self.task = FLAGS.task

        self.out_dim, self.loss_function = self._create_loss()

        self.target_list = self._get_target_list()

        if FLAGS.pairwise_class:
            self.loss_function_pairwise_class = nn.CrossEntropyLoss()

        self.MLPs, self.MLPs_pairwise_class = self._create_decoder_MLPs(D, D, self.target_list, self.out_dim)

        if FLAGS.load_pretrained_GNN and FLAGS.D != 64:
            self.node_embs_proj_to_pretrained_model = MLP(D, 64, activation_type=FLAGS.activation_type,
                                                          num_hidden_lyr=4,
                                                          hidden_channels=[int(D * 0.75), int(D * 0.5), int(D * 0.25),
                                                                           64])

    def _create_ptrans_MLPs(self, dataset, D):
        if dataset is None:
            raise RuntimeError(f'If ptrans is set to True, must send dataset to model init')
        tstype_MLPs = nn.ModuleDict()
        self.seen_pragma_trans_types = dataset.get_attribute('seen_pragma_trans_types')
        for tstype in self.seen_pragma_trans_types:
            if FLAGS.early_late_fusion == 'early':
                if not FLAGS.MLPs_share_weights:
                    for i, _ in enumerate(self.conv_layers):
                        tstype_MLPs[f'{tstype}_{i}'] = MLP(D, D, activation_type=FLAGS.activation)
                else:
                    tstype_MLPs[tstype] = MLP(D, D, activation_type=FLAGS.activation)
            else:
                assert FLAGS.early_late_fusion == 'late'
                tstype_MLPs[tstype] = MLP(D, D, activation_type=FLAGS.activation)
        return tstype_MLPs

    def cal_gae_loss(self, encoded_g, decoded_out):
        target = torch.ones(len(encoded_g),
                            device=FLAGS.device)  ## for similarity, use the negative form for dissimilarity
        target.requires_grad = False
        gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
        return gae_loss

    def forward(self, data, forward_pairwise, tvt=None, epoch=None, iter=None, test_name=None):
        if FLAGS.graph_transformer_option is not None:
            gt_conv_type = FLAGS.graph_transformer_option.get('conv_type', 'mha')
            if gt_conv_type == 'mha':
                x = data.X_padded
            elif gt_conv_type == 'gps_conv':
                x = data.x
            else:
                raise NotImplementedError()
            edge_index, edge_attr, batch, pragmas = data.edge_index, getattr(data, 'edge_attr'), data.batch, getattr(
                data, 'pragmas', None)
        else:
            x, edge_index, edge_attr, batch, pragmas = \
                data.x, data.edge_index, getattr(data, 'edge_attr'), data.batch, getattr(data, 'pragmas', None)

            num_features_node = getattr(self, 'num_features_node')
            if num_features_node is not None:
                if x.shape[1] != num_features_node:
                    saver.log_info(f'Weird error; x.shape={x.shape} while num_features_node={self.num_features_node};')
                    for gname, point in zip(data.gname, data.xy_dict_programl['point']):
                        saver.log_info(f'\t{gname}: {point}')
                    raise RuntimeError(f'Bad input batch; need to debug!')

        gname = None
        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        # print(gname)
        # print(edge_attr.shape)
        outs = []
        out_dict = OrderedDict()

        activation = self._get_act_func()

        if FLAGS.sequence_modeling:
            x_bert, _ = self._apply_bert(x, data)
            x = x_bert

            if FLAGS.data_repr == 'ast' and FLAGS.AST_combine_node_edge_labels:
                x = torch.cat((x_bert, data.X_ast_node_labels), dim=1)

        if FLAGS.sequence_modeling and not FLAGS.apply_act_conv_first:
            act = None
        else:
            act = activation
        out = self._apply_conv_first(self.conv_first, x, edge_index, edge_attr, act)

        outs.append(out)

        out, outs = self._apply_conv_layers(out, self.conv_layers, self.jkn, outs, edge_index, edge_attr, batch, data,
                                            activation)

        if FLAGS.ptrans and FLAGS.early_late_fusion == 'late':
            for tstype in self.seen_pragma_trans_types:
                ts_mlp = self.tstype_MLPs[tstype]
                out = self._ptrans_apply_MLP(out, ts_mlp, data, tstype)

        if FLAGS.graph_transformer_option is not None:
            gt_conv_type = FLAGS.graph_transformer_option.get('conv_type', 'mha')
            if gt_conv_type == 'mha':
                out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths=data.X_padded_lengths, batch_first=True,
                                                              enforce_sorted=False).data
                assert out.shape[0] == data.x.shape[0]
            elif gt_conv_type == 'gps_conv':
                pass
            else:
                raise NotImplementedError()

        node_emb = out
        total_loss = torch.tensor(0.0, device=FLAGS.device)

        if FLAGS.load_pretrained_GNN and FLAGS.D != 64:
            out_dict['node_emb'] = self.node_embs_proj_to_pretrained_model(node_emb)
        else:
            out_dict['node_emb'] = node_emb

        out_dict, out_embed, out_P, out_T = self._gen_graph_level_embedding_from_node(node_emb, batch, data, out_dict,
                                                                                      glob_P=getattr(self, 'glob_P',
                                                                                                     None),
                                                                                      glob_T=getattr(self, 'glob_T',
                                                                                                     None),
                                                                                      glob=getattr(self, 'glob', None))

        # out = global_add_pool(out, batch)
        # out, edge_index, _, batch, perm, score = self.pool1(
        #     out, edge_index, None, batch)
        # ratio = out.size(0) / x.size(0)

        gae_loss = torch.tensor(0.0)
        if FLAGS.gae_T:  # graph auto encoder
            if FLAGS.separate_P_T:
                if FLAGS.pragma_uniform_encoder:
                    gname = 'all'
                encoded_g = self.gae_transform_T[gname](pragmas)
                decoded_out = self.decoder_T(out_dict['emb_T'])
                # gae_loss = self.cal_gae_loss(encoded_g, decoded_out)
                target = torch.ones(len(encoded_g),
                                    device=FLAGS.device)  ## for similarity, use the negative form for dissimilarity
                target.requires_grad = False
                gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
        if FLAGS.gae_P:
            encoded_x = x
            if FLAGS.input_encode:
                encoded_x = self.gate_input(x)
            encoded_g = global_add_pool(encoded_x, batch)  ## simple addition of node embeddings for gae

            if FLAGS.decoder_type == 'None':  ## turn off autograd:
                decoded_out = self.decoder_P(out_dict['emb_P']).detach()
            else:
                decoded_out = self.decoder_P(out_dict['emb_P'])
            # gae_loss = (self.gae_loss_function(encoded_g, decoded_out)).mean()
            gae_loss += self.cal_gae_loss(encoded_g, decoded_out)
            # from saver import saver
            # saver.info(f'cosine similarity is {self.gae_sim_function(encoded_g, decoded_out).mean()}')
            # saver.log_info(f'encoded_g : {F.normalize(encoded_g[0, :], dim=0)}')
            # saver.log_info(f'decoded_out : {F.normalize(decoded_out[0, :], dim=0)}')
        if FLAGS.gae_P or FLAGS.gae_T:
            total_loss += torch.abs(gae_loss)
            # gae_loss = gae_loss.view((len(gae_loss), 1))
            # print(gae_loss.shape)

        loss_dict = OrderedDict()

        if FLAGS.subtask == 'inference' and FLAGS.save_emb:
            d = {'data.gname': data.gname, 'data.key': data.key,
                 'out_embed': out_embed}
            for target_name in self.target_list:
                d[target_name] = get_y_with_target(data, target_name)
            saver.save_emb_accumulate_emb(f'{data.gname}_{data.key}', d, convert_to_np=True)

        self._apply_target_MLPs_with_loss(self.MLPs, out_embed, data, total_loss, out_dict, loss_dict, 'normal')

        if forward_pairwise and FLAGS.pairwise_class and FLAGS.loss_components in ['both', 'class_only']:
            pairwise_comp_results = []
            d1_gemb, d2_gemb = self._split_vec_mat_into_2_halves(out_embed)
            for op in FLAGS.comp_ops:
                if op == 'hadamard':
                    c = d1_gemb * d2_gemb
                elif op == 'diff':
                    c = d1_gemb - d2_gemb
                else:
                    raise NotImplementedError()
                pairwise_comp_results.append(c)
            pairwise_comp_results = torch.cat(pairwise_comp_results, dim=1)

            self._apply_target_MLPs_with_loss(self.MLPs_pairwise_class, pairwise_comp_results, data, total_loss,
                                              out_dict,
                                              loss_dict, 'pairwise_class')

        if FLAGS.itype_mask_perc > 0:
            y = self.mask_MLP_vocab(node_emb)
            y = self.mask_softmax(y)
            mask_loss = self.mask_loss(y, data['itype_true_labels'])
            total_loss += mask_loss
            loss_dict['itype_mask'] = mask_loss

        return out_dict, total_loss, loss_dict, gae_loss

    def _split_vec_mat_into_2_halves(self, input):
        length = input.shape[0]
        assert length % 2 == 0  # divisible by 2 -- otherwise data loader has some issue
        half_point = int(length / 2)
        d1 = input[0:half_point]
        d2 = input[half_point:]
        assert d1.shape == d2.shape
        return d1, d2

    def _ptrans_apply_MLP(self, out, ts_mlp, data, tstype):
        ts_mlp_x = ts_mlp(out)
        ts_nids = getattr(data, tstype)
        out = out + torch.mul(
            ts_mlp_x,
            ts_nids.view(len(ts_mlp_x), 1))
        return out


class MAPE(torch.nn.Module):
    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            if self.quantiles is None:
                assert y_pred.size(
                    -1) == 1, "Prediction should only have one extra dimension"
                y_pred = y_pred[..., 0]
            else:
                y_pred = y_pred.mean(-1)
        return y_pred

    def forward(self, y_pred, target):
        a = (self.to_prediction(y_pred) - target).abs()
        b = (target.abs() + 1e-8)
        loss = a / b
        rtn = torch.mean(loss)
        return rtn


class MSE_WEIGHT_UTIL(torch.nn.Module):
    def forward(self, y_pred, target, target_name):
        loss = ((y_pred - target) ** 2)
        if 'util' in target_name:
            loss = loss * torch.exp(y_pred - 1)
        rtn = torch.mean(loss)
        return rtn


def get_y_with_target(data, target):
    return getattr(data, target.replace('-', '_'))


def _create_gnn(D1, D2):
    if FLAGS.gnn_type == 'gat':
        conv_class = GATConv
    elif FLAGS.gnn_type == 'gcn':
        conv_class = GCNConv
    elif FLAGS.gnn_type == 'transformer':
        conv_class = TransformerConv
    else:
        raise NotImplementedError()

    if FLAGS.gnn_type == 'gcn':
        return conv_class(D1, D2)
    else:
        return conv_class(D1, D2, heads=1)


# def create_model(num_features, itype_vocab_size=None, dataset=None): # used by regression.py
#     return Net(num_features, itype_vocab_size, dataset=dataset)


def feature_extract(model, key_word, gnn_layer=None):
    '''"
        fixes all parameters except for the ones that have "key_word"
        as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                saver.log_info(f'fixing parameter: {name}')
                param.requires_grad = False
            else:
                if 'conv_first' in name or any([f'conv_layers.{d}' in name for d in range(gnn_layer - 1)]):
                    saver.log_info(f'fixing parameter: {name}')
                    param.requires_grad = False

    if FLAGS.random_MLP:
        D = FLAGS.D
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]
        for target in FLAGS.target:
            model.MLPs[target] = MLP(D, 1, activation_type=FLAGS.activation,
                                     hidden_channels=hidden_channels,
                                     num_hidden_lyr=len(hidden_channels))


def check_feature_extract(model, key_word, gnn_layer=None):
    '''"
        checks that all parameters except for the ones that have "key_word" are fixed
        as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                assert param.requires_grad == False
            else:
                if 'conv_first' in name or any([f'conv_layers.{d}' in name for d in range(gnn_layer - 1)]):
                    assert param.requires_grad == False
