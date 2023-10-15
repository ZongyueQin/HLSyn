from data_src_code import get_code_encoder_dim, get_code_encoder_num_layers
from model import Model, get_y_with_target
from model_node_token_interact import create_node_token_interaction_module, create_collaboration, \
    create_graphnorm_layers, create_mlp_afterwards_layers
from data import get_num_features
from saver import saver
from config import FLAGS
from utils_nn import create_graph_att_module
from utils import MLP
from torch_geometric.nn import JumpingKnowledge
import torch
import torch.nn.functional as F
import torch.nn as nn

from collections import OrderedDict


class MultiModalityNet(Model):
    def __init__(self, init_pragma_dict=None, dataset=None):
        super(MultiModalityNet, self).__init__()
        if FLAGS.what_modalities == 'programl+src_code':
            self._init_programl_plus_src_code(dataset)
        else:
            raise NotImplementedError()

    def _init_programl_plus_src_code(self, dataset):
        assert FLAGS.sequence_modeling
        assert not FLAGS.ptrans
        assert not FLAGS.gae_T
        assert not FLAGS.gae_P
        assert not FLAGS.separate_P_T
        assert not FLAGS.pairwise_class
        assert not FLAGS.graph_transformer_option  # TODO: enable it
        assert FLAGS.itype_mask_perc == 0

        num_layers = FLAGS.num_layers
        D = FLAGS.D

        self.task = FLAGS.task

        self.out_dim, self.loss_function = self._create_loss()

        self.target_list = self._get_target_list()

        self.bert_model, self.bert_dim = self._create_src_code_bert()
        self.bert_dim = get_code_encoder_dim()

        conv_class = self._get_GNN_conv_class()

        if FLAGS.combine_fashion == 'share_final_MLPs':
            if FLAGS.encode_edge:
                edge_dim = dataset[0].edge_attr_programl.shape[1]
            else:
                edge_dim = -1

            self.conv_first_programl = self._create_conv_first(edge_dim, dataset[0].x_programl.shape[1], D, conv_class,
                                                               sequence_modeling=False)

            num_features_src_code = get_num_features(dataset, sequence_modeling=True)
            hidden_channels = None
            if FLAGS.feed_p_to_tf and FLAGS.which_pos_to_take == '0_and_1':
                num_features_src_code *= 2
                hidden_channels = [int(num_features_src_code * 0.75), num_features_src_code // 2]
            self.conv_first_src_code = self._create_conv_first(-1, num_features_src_code, D,
                                                               conv_class,
                                                               sequence_modeling=True, hidden_channels=hidden_channels)

            # Separate GNNs and base MLPs in decoder. Just combine eventually.
            self.conv_layers_programl, self.jkn_programl = self._create_GNN_conv_layers(num_layers, edge_dim, D,
                                                                                        conv_class)
            self.conv_layers_src_code, self.jkn_src_code = self._create_GNN_conv_layers(num_layers, None, D,
                                                                                        conv_class)
            self.glob_programl = create_graph_att_module(D)
            self.glob_src_code = create_graph_att_module(D)

            self.base_decoder_programl = MLP(D, D, activation_type=FLAGS.activation_type, num_hidden_lyr=2)
            self.base_decoder_src_code = MLP(D, D, activation_type=FLAGS.activation_type, num_hidden_lyr=2)

            # Graph-level feature fusion at the end.
            input_dim_final_decoder = 2 * D
            hidden_channels = [int(D * 1.5), D, D // 2, D // 4, D // 8]
            if hasattr(FLAGS, 'collaboration_btw_modalities') and FLAGS.collaboration_btw_modalities == 'edge_msgs':
                self.base_decoder_collab = MLP(D, D, activation_type=FLAGS.activation_type, num_hidden_lyr=2)
                input_dim_final_decoder = 3 * D
                hidden_channels = [int(D * 2.5), 2 * D, int(D * 1.5), D, D // 2, D // 4, D // 8]
            self.decoder_shared, _ = self._create_decoder_MLPs(input_dim_final_decoder, D, self.target_list,
                                                               self.out_dim,
                                                               hidden_channels=hidden_channels)

            if hasattr(FLAGS, 'node_token_interaction') and FLAGS.node_token_interaction:
                self.node_token_interaction_GNN = create_node_token_interaction_module(D, get_code_encoder_num_layers())
                if hasattr(FLAGS, 'project_node_embs_before_align') and FLAGS.project_node_embs_before_align:
                    self.node_token_collaboration = create_collaboration(D, get_code_encoder_num_layers())
                    self.node_embs_proj = MLP(D, D, activation_type=FLAGS.activation_type, num_hidden_lyr=2)

                if FLAGS.interleave_GNN_transformer:
                    num_layers_interleaved = get_code_encoder_num_layers()
                    self.interleaved_GNN_layers, self.jkn_programl_interleave = self._create_GNN_conv_layers(
                        num_layers_interleaved + 1, edge_dim, D, conv_class)  # +1 due to -1 in _create_GNN_conv_layers\
                    if hasattr(FLAGS,
                               'collaboration_btw_modalities') and FLAGS.collaboration_btw_modalities == 'edge_msgs':
                        self.jkn_collab_msgs = JumpingKnowledge(FLAGS.jkn_mode, channels=D, num_layers=2)
                    if FLAGS.apply_norm_after_interaction:
                        self.graphnorm_layers_collab = create_graphnorm_layers(D, num_layers_interleaved)
                    if FLAGS.apply_mlp_after_interaction:
                        self.mlp_layers_afterwards = create_mlp_afterwards_layers(D, num_layers_interleaved)

            if FLAGS.pc_links:
                self.alignment_decoder_fine = self._create_alignment_decoder(D)
                self.alignment_decoder_coarse = self._create_alignment_decoder(D)

            if FLAGS.load_pretrained_GNN and FLAGS.D != 64:
                self.node_embs_proj_to_pretrained_model = MLP(D, 64, activation_type=FLAGS.activation_type,
                                                              num_hidden_lyr=4,
                                                              hidden_channels=[int(D * 0.75), int(D * 0.5),
                                                                               int(D * 0.25), 64])



        elif FLAGS.combine_fashion == 'share_GNNs_MLPs':
            if FLAGS.encode_edge:
                edge_dim = dataset[0].edge_attr_shared.shape[1]
            else:
                edge_dim = -1

            self.conv_first_programl = MLP(dataset[0].x_programl.shape[1], D, activation_type=FLAGS.activation,
                                           num_hidden_lyr=2)
            self.conv_first_src_code = self._create_conv_first(-1, get_num_features(dataset, sequence_modeling=True), D,
                                                               conv_class,
                                                               sequence_modeling=True)

            self.conv_layers, self.jkn = self._create_GNN_conv_layers(num_layers, edge_dim, D, conv_class)
            self.glob = create_graph_att_module(D)
            input_to_final_decoder_dim = D
            if FLAGS.multi_glevel_embs:
                self.glob_programl = create_graph_att_module(D)
                self.glob_src_code = create_graph_att_module(D)
                self.base_decoder_programl = MLP(D, D, activation_type=FLAGS.activation_type, num_hidden_lyr=2)
                self.base_decoder_src_code = MLP(D, D, activation_type=FLAGS.activation_type, num_hidden_lyr=2)
                self.base_decoder_all = MLP(D, D, activation_type=FLAGS.activation_type, num_hidden_lyr=2)
                input_to_final_decoder_dim = 3 * D
                self.decoder_shared, _ = self._create_decoder_MLPs(input_to_final_decoder_dim, D, self.target_list,
                                                                   self.out_dim,
                                                                   hidden_channels=[int(D * 2.5), int(D * 2),
                                                                                    int(D * 1.5), D, D // 2, D // 4,
                                                                                    D // 8])
            else:
                self.decoder_shared, _ = self._create_decoder_MLPs(input_to_final_decoder_dim, D, self.target_list,
                                                                   self.out_dim)

            assert not FLAGS.load_pretrained_GNN
        else:
            raise NotImplementedError()

    def forward(self, data, forward_pairwise, tvt=None, epoch=None, iter=None, test_name=None):

        # Turn on below for quick debugging, e.g. the data loading/batching.
        # ld = {}
        # od = {}
        # for target_name in self.target_list:
        #     ld[target_name] = torch.tensor(0.0)
        #     od[target_name] = get_y_with_target(data, target_name)
        # return od, torch.tensor(0.0), ld, torch.tensor(0.0)

        if FLAGS.what_modalities == 'programl+src_code':
            return self._forward_programl_plus_src_code(data, tvt)
        else:
            raise NotImplementedError()

    def _forward_programl_plus_src_code(self, data, tvt):
        assert FLAGS.data_repr != 'ast'

        outs = []
        outs_programl = []
        outs_src_code = []
        out_dict = OrderedDict()
        total_loss = torch.tensor(0.0, device=FLAGS.device)
        loss_dict = OrderedDict()

        activation = self._get_act_func()

        edge_index_programl, edge_attr_programl, edge_index_src_code, edge_attr_src_code = None, None, None, None
        # batch_programl, batch_src_code = None, None
        edge_index_shared, edge_attr_shared = None, None
        batch_shared = None
        if FLAGS.combine_fashion == 'share_final_MLPs':  # TODO handle edge_attr_src_code
            x_programl, edge_index_programl, edge_attr_programl, batch_programl, \
            x_src_code, edge_index_src_code, edge_attr_src_code, batch_src_code, \
            pragmas = \
                data.x_programl, data.edge_index_programl, getattr(data, 'edge_attr_programl'), data.x_programl_batch, \
                data.x_src_code, data.edge_index_src_code, getattr(data, 'edge_attr_src_code',
                                                                   None), data.x_src_code_batch, \
                getattr(data, 'pragmas', None)
        elif FLAGS.combine_fashion == 'share_GNNs_MLPs':
            x_programl, x_src_code, edge_index_shared, edge_attr_shared, batch_shared, pragmas = \
                data.x_programl, data.x_src_code, data.edge_index_shared, getattr(data,
                                                                                  'edge_attr_shared'), data.x_dummy_batch, \
                getattr(data, 'pragmas', None)
            batch_programl, batch_src_code = data.x_programl_batch, data.x_src_code_batch
            assert x_programl.shape[0] + x_src_code.shape[0] == batch_shared.shape[0] == data.x_dummy.shape[0]
        else:
            raise NotImplementedError()

        if FLAGS.apply_act_conv_first:
            act = activation
        else:
            act = None

        # out_src_code = None
        # x_src_code_full = None

        # Always do programl encoding first.
        out_programl = self._apply_conv_first(self.conv_first_programl, x_programl, edge_index_programl,
                                              edge_attr_programl,
                                              act)

        outs_programl.append(out_programl)

        if FLAGS.combine_fashion == 'share_final_MLPs':
            out_programl, outs_programl = self._apply_conv_layers(out_programl, self.conv_layers_programl,
                                                                  self.jkn_programl,
                                                                  outs_programl, edge_index_programl,
                                                                  edge_attr_programl, batch_programl,
                                                                  data, activation)

            out_dict['node_emb'] = out_programl

            out_dict, out_embed_programl, *_ = self._gen_graph_level_embedding_from_node(out_programl, batch_programl,
                                                                                         data,
                                                                                         out_dict,
                                                                                         glob_P=None,
                                                                                         glob_T=None,
                                                                                         glob=self.glob_programl)
            out_embed_programl = self.base_decoder_programl(out_embed_programl)

            nti_objs = None
            out_programl_proj = out_programl
            if hasattr(FLAGS, 'node_token_interaction') and FLAGS.node_token_interaction:
                if not FLAGS.interleave_GNN_transformer:
                    # tricky: project before feeding into interaction
                    if hasattr(FLAGS, 'project_node_embs_before_align') and FLAGS.project_node_embs_before_align:
                        out_programl_proj = self.node_embs_proj(out_programl)
                nti_objs = {'out_programl': out_programl_proj,
                            'node_token_interaction_GNN': self.node_token_interaction_GNN,
                            'data': data, 'training': self.training}
                if hasattr(FLAGS, 'collaboration_btw_modalities') and FLAGS.collaboration_btw_modalities == 'edge_msgs':
                    nti_objs['node_token_collaboration'] = self.node_token_collaboration

                if hasattr(FLAGS, 'interleave_GNN_transformer') and FLAGS.interleave_GNN_transformer:
                    nti_objs['interleaved_GNN_layers'] = self.interleaved_GNN_layers
                    nti_objs['edge_index_programl'] = edge_index_programl
                    nti_objs['edge_attr_programl'] = edge_attr_programl
                    nti_objs['act'] = activation
                    nti_objs['jkn_programl_interleave'] = self.jkn_programl_interleave
                    if FLAGS.apply_norm_after_interaction:
                        nti_objs['graphnorm_layers_collab'] = self.graphnorm_layers_collab
                    if FLAGS.apply_mlp_after_interaction:
                        nti_objs['mlp_layers_afterwards'] = self.mlp_layers_afterwards

            # Encode source code.
            if not FLAGS.feed_p_to_tf:
                # Normal source code embedding using code transformer without fancy feeding.
                # However, can still do node-token interaction which is fancy.
                x_bert, x_src_code_full = self._apply_bert(x_src_code, data, nti_objs=nti_objs)
                out_src_code = self._apply_conv_first(self.conv_first_src_code, x_bert, edge_index_src_code,
                                                      edge_attr_src_code,
                                                      act)
                outs_src_code.append(out_src_code)

            else:
                assert out_embed_programl.shape[1] == self.bert_dim, \
                    f'out_embed_programl.shape={out_embed_programl.shape}; self.bert_dim={self.bert_dim}'
                out_embed_programl_repeat = out_embed_programl.repeat_interleave(data.num_chunks, dim=0)
                assert out_embed_programl_repeat.shape == (x_src_code.shape[0], self.bert_dim)
                tot_num_chunks = torch.sum(data.num_chunks)
                out_embed_programl_repeat = out_embed_programl_repeat.view(tot_num_chunks, 1, self.bert_dim)
                _, x_src_code_full = self._apply_bert(x_src_code, data, additional_x=out_embed_programl_repeat,
                                                      nti_objs=nti_objs)
                x_bert = x_src_code_full['last_hidden_state']
                if FLAGS.which_pos_to_take == '0':
                    x_bert = x_bert[:, 0, :]
                    assert x_bert.shape == (tot_num_chunks, self.bert_dim)
                elif FLAGS.which_pos_to_take == '1':
                    x_bert = x_bert[:, 1, :]
                    assert x_bert.shape == (tot_num_chunks, self.bert_dim)
                elif FLAGS.which_pos_to_take == '0_and_1':
                    x_bert = x_bert[:, 0:2, :]
                    assert x_bert.shape == (tot_num_chunks, 2, self.bert_dim)
                    x_bert = x_bert.view(tot_num_chunks, 2 * self.bert_dim)
                else:
                    raise NotImplementedError()

                out_src_code = self._apply_conv_first(self.conv_first_src_code, x_bert, edge_index_src_code,
                                                      edge_attr_src_code,
                                                      act)
                outs_src_code.append(out_src_code)

            out_src_code, outs_src_code = self._apply_conv_layers(out_src_code, self.conv_layers_src_code,
                                                                  self.jkn_src_code,
                                                                  outs_src_code, edge_index_src_code,
                                                                  edge_attr_src_code, batch_src_code,
                                                                  data, activation)
            out_dict, out_embed_src_code, *_ = self._gen_graph_level_embedding_from_node(out_src_code, batch_src_code,
                                                                                         data,
                                                                                         out_dict,
                                                                                         glob_P=None,
                                                                                         glob_T=None,
                                                                                         glob=self.glob_src_code)
            out_embed_src_code = self.base_decoder_src_code(out_embed_src_code)

            if hasattr(FLAGS, 'collaboration_btw_modalities') and FLAGS.collaboration_btw_modalities == 'edge_msgs':

                edge_msgs_aggregated_list = nti_objs['edge_msgs_aggregated_list']
                if FLAGS.interleave_GNN_transformer:
                    edge_msgs_aggregated = self.jkn_collab_msgs(edge_msgs_aggregated_list)

                    # tricky: project after feeding into interaction and re-generate the graph-level embedding!
                    if hasattr(FLAGS, 'project_node_embs_before_align') and FLAGS.project_node_embs_before_align:
                        out_programl_proj = self.node_embs_proj(nti_objs['out_programl'])

                        out_dict, out_embed_programl, *_ = self._gen_graph_level_embedding_from_node(out_programl_proj,
                                                                                                     batch_programl,
                                                                                                     data,
                                                                                                     out_dict,
                                                                                                     glob_P=None,
                                                                                                     glob_T=None,
                                                                                                     glob=self.glob_programl)
                        out_embed_programl = self.base_decoder_programl(out_embed_programl)

                else:
                    assert len(edge_msgs_aggregated_list) == 1
                    edge_msgs_aggregated = edge_msgs_aggregated_list[0]

                edge_msgs = self.base_decoder_collab(edge_msgs_aggregated)
                input_to_final_MLPs = torch.cat((out_embed_programl, out_embed_src_code, edge_msgs), dim=1)  # fusion
            else:
                input_to_final_MLPs = torch.cat((out_embed_programl, out_embed_src_code), dim=1)  # fusion

            if FLAGS.subtask == 'inference' and FLAGS.save_emb:
                d = {'data.gname': data.gname, 'data.key': data.key,
                     'out_embed_programl': out_embed_programl,
                     'out_embed_src_code': out_embed_src_code}
                for target_name in self.target_list:
                    d[target_name] = get_y_with_target(data, target_name)
                saver.save_emb_accumulate_emb(f'{data.gname}_{data.key}', d, convert_to_np=True)

            self._apply_target_MLPs_with_loss(self.decoder_shared, input_to_final_MLPs, data, total_loss, out_dict,
                                              loss_dict, 'normal')

            out_dict['node_emb'] = out_programl_proj

            if FLAGS.load_pretrained_GNN and FLAGS.D != 64:
                out_dict['node_emb'] = self.node_embs_proj_to_pretrained_model(out_programl_proj)

            if FLAGS.pc_links:
                if FLAGS.node_token_alignment_loss:

                    x_node_programl = out_programl_proj
                    x_token_src_code = x_src_code_full['last_hidden_state']

                    # x_node_programl_pos = x_node_programl[[5, 7, 10]]
                    # x_token_src_code_pos = x_token_src_code[[0, 0, 0], [0, 0, 0]]
                    # num_ids_pc_loss = 3

                    # This is the loss function, which would not be needed in testing.
                    if FLAGS.pc_links_holdout_ratio == 0:
                        pos_node_ids = data.pos_node_ids
                        pos_token_ids = data.pos_token_ids
                        pos_chunk_ids = data.pos_chunk_ids
                    else:
                        pos_node_ids = data.pos_node_ids_test
                        pos_token_ids = data.pos_token_ids_test
                        pos_chunk_ids = data.pos_chunk_ids_test

                    x_node_programl_pos = x_node_programl[pos_node_ids]
                    if FLAGS.feed_p_to_tf:
                        pos_token_ids += 1  # TODO: pay attention to this tricky code: shift by 1 when feed
                    x_token_src_code_pos = x_token_src_code[pos_chunk_ids, pos_token_ids]
                    num_ids_pc_loss = len(pos_chunk_ids)

                    tot_num_nodes, d1 = x_node_programl.shape
                    tot_num_chunks, seq_len_ie_num_tokens, d2 = x_token_src_code.shape
                    assert d1 == d2

                    # Tricky: Each node may corredpond to multiple tokens,
                    # i.e. (node 0, chunk 0, token 10), (node 0, chunk 1, token 5),
                    #      (node 1, chunk 0, token 20), etc.
                    neg_node_ids = torch.randint(tot_num_nodes, (num_ids_pc_loss,))
                    neg_chunk_ids = torch.randint(tot_num_chunks, (num_ids_pc_loss,))
                    neg_token_ids = torch.randint(seq_len_ie_num_tokens, (num_ids_pc_loss,))

                    x_node_programl_neg = out_programl_proj[neg_node_ids]
                    x_token_src_code_neg = x_token_src_code[neg_chunk_ids, neg_token_ids]

                    assert x_node_programl_pos.shape == x_token_src_code_pos.shape == x_node_programl_neg.shape == x_token_src_code_neg.shape

                    pred_pos = self._alignment_decode(x_node_programl_pos, x_token_src_code_pos,
                                                      self.alignment_decoder_fine)
                    pred_neg = self._alignment_decode(x_node_programl_neg, x_token_src_code_neg,
                                                      self.alignment_decoder_fine)

                    loss_pos = \
                        F.binary_cross_entropy_with_logits(
                            pred_pos, torch.ones_like(pred_pos, dtype=torch.float, device=FLAGS.device))
                    loss_neg = \
                        F.binary_cross_entropy_with_logits(
                            pred_neg, torch.zeros_like(pred_neg, dtype=torch.float, device=FLAGS.device))
                    loss_pc_links = (loss_pos + loss_neg) / 2
                    total_loss += loss_pc_links
                    loss_dict['loss_pc_links'] = loss_pc_links
                    # print(loss_pc_links, loss_pc_links)

                if FLAGS.gs_alignment_loss:
                    num_data1, D1 = out_embed_programl.shape
                    num_data2, D2 = out_embed_src_code.shape
                    assert num_data1 == num_data2 and D1 == D2
                    num_data = num_data1
                    if FLAGS.batch_size == 1 or num_data1 == 1:
                        saver.log_info_once(
                            f'FLAGS.batch_size={FLAGS.batch_size}, num_data={num_data} -- skip gs_alignment_loss')
                        loss_gs_alignment = torch.tensor(0.0)
                    else:
                        neg_ids_programl, neg_ids_src_code = self._gen_neg_ids_for_gs_alignment_loss(num_data)
                        gs_pred_pos = self._alignment_decode(out_embed_programl, out_embed_src_code,
                                                             self.alignment_decoder_coarse)
                        gs_pred_neg = self._alignment_decode(out_embed_programl[neg_ids_programl],
                                                             out_embed_src_code[neg_ids_src_code],
                                                             self.alignment_decoder_coarse)
                        gs_loss_pos = \
                            F.binary_cross_entropy_with_logits(
                                gs_pred_pos, torch.ones_like(gs_pred_pos, dtype=torch.float, device=FLAGS.device))
                        gs_loss_neg = \
                            F.binary_cross_entropy_with_logits(
                                gs_pred_neg, torch.zeros_like(gs_pred_neg, dtype=torch.float, device=FLAGS.device))
                        loss_gs_alignment = (gs_loss_pos + gs_loss_neg) / 2
                    total_loss += loss_gs_alignment
                    loss_dict['loss_gs_alignment'] = loss_gs_alignment


        elif FLAGS.combine_fashion == 'share_GNNs_MLPs':
            assert FLAGS.load_encoders == 'None' or FLAGS.load_encoders is None, 'Do not support special encoders since encoder only works with CDFG modality alone'

            x_bert, x_src_code_full = self._apply_bert(x_src_code, data)
            out_src_code = self._apply_conv_first(self.conv_first_src_code, x_bert, edge_index_src_code,
                                                  edge_attr_src_code,
                                                  act)
            outs_src_code.append(out_src_code)

            '''The below code is WRONG. Very tricky bug!'''
            # input_to_GNNs = torch.cat((out_programl, out_src_code), dim=0)  # fusion; vertically: programl then src_code
            '''Should do the following instead,'''
            input_to_GNNs, mask_programl, mask_src_code = self._merge_to_get_input_to_shared_GNNs(out_programl,
                                                                                                  out_src_code,
                                                                                                  batch_programl,
                                                                                                  batch_src_code)
            out, outs = self._apply_conv_layers(input_to_GNNs, self.conv_layers,
                                                self.jkn,
                                                outs, edge_index_shared,
                                                edge_attr_shared, batch_shared,
                                                data, activation)
            out_dict, out_embed, *_ = self._gen_graph_level_embedding_from_node(out, batch_shared,
                                                                                data,
                                                                                out_dict,
                                                                                glob_P=None,
                                                                                glob_T=None,
                                                                                glob=self.glob)

            input_to_final_MLPs = out_embed
            if FLAGS.multi_glevel_embs:
                out_dict, out_embed_programl, *_ = self._gen_graph_level_embedding_from_node(out, batch_shared,
                                                                                             data,
                                                                                             out_dict,
                                                                                             glob_P=None,
                                                                                             glob_T=None,
                                                                                             glob=self.glob_programl,
                                                                                             att_mask=mask_programl)
                out_dict, out_embed_src_code, *_ = self._gen_graph_level_embedding_from_node(out, batch_shared,
                                                                                             data,
                                                                                             out_dict,
                                                                                             glob_P=None,
                                                                                             glob_T=None,
                                                                                             glob=self.glob_src_code,
                                                                                             att_mask=mask_src_code)

                out_embed_programl = self.base_decoder_programl(out_embed_programl)
                out_embed_src_code = self.base_decoder_src_code(out_embed_src_code)
                out_embed_all = self.base_decoder_all(out_embed)

                input_to_final_MLPs = torch.cat((out_embed_programl, out_embed_src_code, out_embed_all), dim=1)

            self._apply_target_MLPs_with_loss(self.decoder_shared, input_to_final_MLPs, data, total_loss, out_dict,
                                              loss_dict,
                                              'normal')

        else:
            raise NotImplementedError()

        return out_dict, total_loss, loss_dict, torch.tensor(0.0)

    def _merge_to_get_input_to_shared_GNNs(self, out_programl, out_src_code, batch_programl, batch_src_code):
        chunk_sizes_programl = self._get_chunk_sizes(batch_programl)
        chunk_sizes_src_code = self._get_chunk_sizes(batch_src_code)
        list_of_tensors_programl = torch.split(out_programl, chunk_sizes_programl)
        list_of_tensors_src_code = torch.split(out_src_code, chunk_sizes_src_code)
        assert len(list_of_tensors_programl) == len(list_of_tensors_src_code) == FLAGS.batch_size
        to_merge = []
        mask_programl = []
        mask_src_code = []
        for i in range(FLAGS.batch_size):
            p = list_of_tensors_programl[i]
            s = list_of_tensors_src_code[i]
            to_merge.append(p)  # programl first
            to_merge.append(s)  # then src code
            if FLAGS.multi_glevel_embs:
                mask_programl.append(torch.tensor([0] * p.shape[0], device=FLAGS.device))
                mask_programl.append(torch.tensor([-10000] * s.shape[0], device=FLAGS.device))
                mask_src_code.append(torch.tensor([-10000] * p.shape[0], device=FLAGS.device))
                mask_src_code.append(torch.tensor([0] * s.shape[0], device=FLAGS.device))
        rtn = torch.cat(to_merge, dim=0)  # vertical
        if FLAGS.multi_glevel_embs:
            mask_programl = torch.cat(mask_programl, dim=0).view(-1, 1)
            mask_src_code = torch.cat(mask_src_code, dim=0).view(-1, 1)
            assert mask_programl.shape == mask_src_code.shape == (rtn.shape[0], 1)
        assert rtn.shape == (out_programl.shape[0] + out_src_code.shape[0], out_programl.shape[1])
        return rtn, mask_programl, mask_src_code

    def _get_chunk_sizes(self, batch):
        sizes = []
        for i in range(FLAGS.batch_size):
            size = (batch == i).sum(dim=0).item()
            sizes.append(size)

        assert (batch == FLAGS.batch_size).sum(dim=0).item() == 0
        return sizes

    def _gen_neg_ids_for_gs_alignment_loss(self, num_data):
        neg_ids_programl, neg_ids_src_code = None, None
        done = False
        iter = 0
        while not done:
            neg_ids_programl = torch.randperm(num_data)
            neg_ids_src_code = torch.randperm(num_data)
            done = True
            for id1, id2 in zip(neg_ids_programl, neg_ids_src_code):
                if id1 == id2:
                    done = False
                    break
            iter += 1
            if iter >= 20:
                saver.log_info(f'_gen_neg_ids_for_gs_alignment_loss: iter {iter}; num_data={num_data}')
        return neg_ids_programl, neg_ids_src_code

    def _create_alignment_decoder(self, D):
        if hasattr(FLAGS, 'alignment_decoder'):
            if FLAGS.alignment_decoder == 'dot':
                return None
            elif FLAGS.alignment_decoder == 'cosine':
                return nn.CosineSimilarity(dim=1)
            elif FLAGS.alignment_decoder == 'concat_MLP':
                return MLP(2 * D, 1, activation_type=FLAGS.activation_type, num_hidden_lyr=9,
                           hidden_channels=[int(D * 1.5), D, D // 2, D // 4,
                                            D // 8, 32, 16, 8, 1])
            else:
                raise NotImplementedError()

    def _alignment_decode(self, x1, x2, align_decoder):
        if hasattr(FLAGS, 'alignment_decoder'):
            if FLAGS.alignment_decoder == 'dot':
                pred = torch.sum(x1 * x2, dim=1)
            elif FLAGS.alignment_decoder == 'cosine':
                pred = align_decoder(x1, x2)
            elif FLAGS.alignment_decoder == 'concat_MLP':
                pred = align_decoder(torch.cat((x1, x2), dim=1))
            else:
                raise NotImplementedError()
        else:
            pred = torch.sum(x1 * x2, dim=1)
        return pred
