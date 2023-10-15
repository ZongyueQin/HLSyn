from utils_nn import create_graph_att_module, LpModule
from config import FLAGS
from our_transformer_conv import OurTransformerConv
from utils import MLP
from saver import saver

from torch_geometric.nn.norm import GraphNorm

import torch
import torch.nn as nn


def create_node_token_interaction_module(D, code_encoder_num_layers):
    def _create_bridge_GNN():
        return OurTransformerConv(D, D, edge_dim=edge_dim, root_weight=False)

    if FLAGS.pc_links_aug in ['all_line_sw', 's_grease']:
        edge_dim = 2
    elif FLAGS.pc_links_aug == 'all_line_swp':
        edge_dim = 3
    elif FLAGS.pc_links_aug == 'all_line_swp_grease':
        edge_dim = 4
    else:
        edge_dim = None

    if FLAGS.interleave_GNN_transformer and not FLAGS.weight_tying_for_interaction_GNN:
        bridge_GNN_layers = nn.ModuleList()
        num_bridge_GNNs = code_encoder_num_layers
        if FLAGS.interact_after_all_layers:
            num_bridge_GNNs += 1
        for _ in range(num_bridge_GNNs):
            bridge_GNN_layers.append(_create_bridge_GNN())
        return bridge_GNN_layers
    else:
        return _create_bridge_GNN()


def _get_bridge_GNN_layer(nti_objs, layer_id):
    if FLAGS.interleave_GNN_transformer and not FLAGS.weight_tying_for_interaction_GNN:
        assert 0 <= layer_id < len(nti_objs['node_token_interaction_GNN'])
        return nti_objs['node_token_interaction_GNN'][layer_id]  # index into a list of modules
    else:
        return nti_objs['node_token_interaction_GNN']  # just one module


def create_collaboration(D, code_encoder_num_layers):
    if FLAGS.interleave_GNN_transformer and not FLAGS.weight_tying_for_interaction_GNN:
        collaboration_layers = nn.ModuleList()
        num_collab_layers = code_encoder_num_layers
        if FLAGS.interact_after_all_layers:
            num_collab_layers += 1
        for _ in range(num_collab_layers):
            collaboration_layers.append(create_graph_att_module(D))
        return collaboration_layers
    else:
        return create_graph_att_module(D)


def _get_collaboration_layer(nti_objs, layer_id):
    if FLAGS.interleave_GNN_transformer and not FLAGS.weight_tying_for_interaction_GNN:
        assert 0 <= layer_id < len(nti_objs['node_token_collaboration']) == len(nti_objs['node_token_interaction_GNN'])
        return nti_objs['node_token_collaboration'][layer_id]  # index into a list of modules
    else:
        return nti_objs['node_token_collaboration']  # just one module


def create_graphnorm_layers(D, code_encoder_num_layers):
    def _get_norm(norm_name, dim):
        if norm_name == 'layer':
            return torch.nn.LayerNorm(dim)
        elif norm_name == 'graph':
            return GraphNorm(dim)
        elif norm_name == 'L2':
            return LpModule(2)
        elif norm_name == 'L1':
            return LpModule(1)
        else:
            raise NotImplementedError(f'{norm_name}')

    rtn = nn.ModuleList()
    use = code_encoder_num_layers
    if FLAGS.interact_after_all_layers:
        use = code_encoder_num_layers + 1
    for _ in range(use):
        rtn.append(_get_norm(FLAGS.emb_norm_method, D))
    return rtn


def _get_graphnorm_layer(nti_objs, layer_id):
    assert 0 <= layer_id < len(nti_objs['graphnorm_layers_collab'])
    return nti_objs['graphnorm_layers_collab'][layer_id]  # index into a list of modules


def create_mlp_afterwards_layers(D, code_encoder_num_layers):
    rtn = nn.ModuleList()
    use = code_encoder_num_layers
    if FLAGS.interact_after_all_layers:
        use = code_encoder_num_layers + 1
    for _ in range(use):
        rtn.append(MLP(D, D, activation_type=FLAGS.activation_type, num_hidden_lyr=2))
    return rtn


def _get_mlp_afterwards_layer(nti_objs, layer_id):
    assert 0 <= layer_id < len(nti_objs['mlp_layers_afterwards'])
    return nti_objs['mlp_layers_afterwards'][layer_id]  # index into a list of modules


def interact_the_two_modalities(inputs_embeds, nti_objs, layer_id):
    '''
    Handles the interaction between programl graph and source code sequence.
    Perform intra-graph GNN for programl and interaction across the two.
    The transformer applied to source code sequence is done by the caller of this function in our_codet5.
    '''
    if FLAGS.code_encoder == 'codet5':
        assert 0 <= layer_id <= 6
    else:
        raise NotImplementedError()

    if hasattr(FLAGS, 'node_token_interaction') and FLAGS.node_token_interaction:

        # Run (intra-modality/programl) GNN (possibly).
        if FLAGS.interleave_GNN_transformer and layer_id > 0:  # no need to run GNN for the very initial interaction

            if 'outs' not in nti_objs:
                nti_objs['outs'] = []
                nti_objs['outs'].append(nti_objs['out_programl'])

            assert FLAGS.graph_transformer_option is None, 'not supported yet'
            GNN_layer_id = layer_id - 1
            GNN_layer = nti_objs['interleaved_GNN_layers'][GNN_layer_id]
            nti_objs['out_programl'] = GNN_layer(
                nti_objs['out_programl'], edge_index=nti_objs['edge_index_programl'],
                edge_attr=nti_objs['edge_attr_programl'])

            if GNN_layer_id != len(nti_objs['interleaved_GNN_layers']) - 1:
                nti_objs['out_programl'] = nti_objs['act'](nti_objs['out_programl'])
            nti_objs['outs'].append(nti_objs['out_programl'])

        # Interact (possibly).
        if layer_id == 0:
            do_interaction = True
        else:
            if FLAGS.interleave_GNN_transformer:
                do_interaction = True
                if layer_id == 6:
                    # Special for after the last layer.
                    if FLAGS.jkn_enable:
                        # This line updates out_programl which will be used by the outside.
                        nti_objs['out_programl'] = nti_objs['jkn_programl_interleave'](nti_objs['outs'])
                    do_interaction = FLAGS.interact_after_all_layers
            else:
                do_interaction = False
        if do_interaction:
            inputs_embeds = _forward_node_token_interaction(inputs_embeds, nti_objs,
                                                            layer_id)  # TODO: both directions msg passing --> ie update nti_objs['out_programl']; use the right bridge-GNN; TODO: apply proj MLP to node & token embds for alignment????? check final alignment loss for which node embeddings are taken

    return inputs_embeds


def _forward_node_token_interaction(inputs_embeds_src_code, nti_objs, layer_id):
    out_programl = nti_objs['out_programl']
    num_chunks, num_tokens, embed_dim = inputs_embeds_src_code.shape
    inputs_embeds_src_code_flat = inputs_embeds_src_code.view(num_chunks * num_tokens, embed_dim)

    num_nodes_total = out_programl.shape[0]
    # num_tokens_total = inputs_embeds_src_code_flat.shape[0]
    x = torch.cat((out_programl, inputs_embeds_src_code_flat), dim=0)

    edge_msg_batch = None
    if (FLAGS.pc_links_holdout_ratio > 0 and nti_objs['training']) or \
            (hasattr(FLAGS, 'pc_links_force_use_train_when_inference')
             and FLAGS.pc_links_force_use_train_when_inference):
        pos_node_ids = nti_objs['data'].pos_node_ids_train
        pos_token_global_ids_in_chunks = nti_objs['data'].pos_token_global_ids_in_chunks_train
        if hasattr(FLAGS, 'collaboration_btw_modalities') and FLAGS.collaboration_btw_modalities == 'edge_msgs':
            edge_msg_batch = nti_objs['data'].pos_node_ids_train_batch
        saver.log_info_once(f'model_node_token_interact.py: Use pos_token_global_ids_in_chunks_train')
    else:
        pos_node_ids = nti_objs['data'].pos_node_ids
        pos_token_global_ids_in_chunks = nti_objs['data'].pos_token_global_ids_in_chunks
        if hasattr(FLAGS, 'collaboration_btw_modalities') and FLAGS.collaboration_btw_modalities == 'edge_msgs':
            edge_msg_batch = nti_objs['data'].pos_node_ids_batch
    node_ids = pos_node_ids.view(1, -1)
    # Because cat (<nodes>, <tokens>) (see above), need to increment the token ids by number of nodes
    # in order to get the right id.
    shifted_token_ids = pos_token_global_ids_in_chunks.view(1, -1) + num_nodes_total
    edge_index = torch.cat((node_ids, shifted_token_ids), dim=0)  # from node --> token; no other direction needed

    # edge_index = torch.LongTensor([(0, 1)]).t().contiguous()
    if FLAGS.pc_links_aug in ['all_line_sw', 'all_line_swp', 'all_line_swp_grease', 's_grease']:
        if nti_objs['training'] and FLAGS.pc_links_holdout_ratio > 0:
            edge_attr = nti_objs['data'].pos_token_sw_type_in_chunks_train
        else:
            edge_attr = nti_objs['data'].pos_token_sw_type_in_chunks
    else:
        edge_attr = None

    if FLAGS.interleave_GNN_transformer:
        # Bi-directional information ezchange across the two modalities.
        edge_index_token_to_node = torch.cat((shifted_token_ids, node_ids), dim=0)
        edge_index = torch.cat((edge_index, edge_index_token_to_node), dim=1)
        if FLAGS.pc_links_aug not in ['grease', None, 'all_line']:
            edge_attr = torch.cat((edge_attr, edge_attr), dim=0)  # edge for the other direction has the same edge type

    # Bridge-GNN.
    bridge_GNN = _get_bridge_GNN_layer(nti_objs, layer_id)
    x_output = bridge_GNN(x, edge_index, edge_attr=edge_attr)

    if hasattr(FLAGS, 'collaboration_btw_modalities') and FLAGS.collaboration_btw_modalities == 'edge_msgs':
        if 'edge_msgs_aggregated_list' not in nti_objs:
            nti_objs['edge_msgs_aggregated_list'] = []
        edge_msgs = bridge_GNN.edge_msgs_saved_in_propagation
        assert edge_msgs.shape[1] == 1
        edge_msgs = edge_msgs.view(edge_msgs.shape[0], edge_msgs.shape[2])
        collab_layer = _get_collaboration_layer(nti_objs, layer_id)
        if FLAGS.interleave_GNN_transformer:
            edge_msg_batch = torch.cat((edge_msg_batch, edge_msg_batch))
        edge_msgs_aggregated, _ = collab_layer(edge_msgs, edge_msg_batch)
        nti_objs['edge_msgs_aggregated_list'].append(edge_msgs_aggregated)
    # The following two lines should return identical results, indicating how many tokens are actually updated.
    # set(nti_objs['data'].pos_token_global_ids_in_chunks.tolist())
    # sum([1 if x > 0 else 0 for x in (x_output.numpy() != 0).sum(1)])

    if hasattr(FLAGS, 'actually_aggregate') and not FLAGS.actually_aggregate:
        x_new = x
    else:
        x_new = x + x_output

    if FLAGS.interleave_GNN_transformer:
        x_programl = x_new[0:num_nodes_total]
        nti_objs['out_programl'] = x_programl  # critical: need to let later layers use the updated node embeddings
        if FLAGS.apply_norm_after_interaction:
            graphnorm = _get_graphnorm_layer(nti_objs, layer_id)
            assert len(nti_objs['data'].x_programl_batch) == x_programl.shape[0]
            nti_objs['out_programl'] = graphnorm(nti_objs['out_programl'], batch=nti_objs['data'].x_programl_batch)
        if FLAGS.apply_mlp_after_interaction:
            mlp = _get_mlp_afterwards_layer(nti_objs, layer_id)
            out_programl_temp = mlp(nti_objs['out_programl'])
            nti_objs['out_programl'] = nti_objs[
                                           'out_programl'] + out_programl_temp  # similar to feed forward in transformer

    x_src_code = x_new[num_nodes_total:]
    x_src_code = x_src_code.view(num_chunks, num_tokens, embed_dim)
    assert x_src_code.shape == inputs_embeds_src_code.shape
    return x_src_code
