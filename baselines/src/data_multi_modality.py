from config import FLAGS
from data_programl import init_preprocessors_programl, read_programl_graph, encode_feat_dict_programl, \
    fit_preprocessors_programl, encode_X_torch_programl
from data_src_code import init_preprocessors_src_code, read_source_code, encode_feat_dict_src_code, \
    fit_preprocessors_src_code, encode_X_torch_src_code
from saver import saver
from torch_geometric.data import Data
import torch
from collections import OrderedDict


class MultiModalityData(object):
    def __init__(self, g_programl, g_src_code):
        self.g_programl = g_programl
        self.g_src_code = g_src_code


class SeparateGNNData(Data):
    def __init__(self, edge_index_programl, x_programl, edge_index_src_code, x_src_code,
                 edge_attr_programl, **kwargs):
        super().__init__()
        self.edge_index_programl = edge_index_programl
        self.x_programl = x_programl
        self.edge_index_src_code = edge_index_src_code
        self.x_src_code = x_src_code
        self.edge_attr_programl = edge_attr_programl
        # self.attention_mask = attention_mask
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_programl':
            return self.x_programl.size(0)
        elif key == 'edge_index_src_code':
            return self.x_src_code.size(0)

        # elif key == 'edge_attr_programl': # buggy code! shouldn't be incremented!
        #     return self.x_programl.size(0)
        # elif key == 'attention_mask':
        #     return self.x_src_code.size(0)

        # Note: Below is critical for handling programl to source code links. The node ids and chunk ids must be
        # shifted by the correct number of nodes and tokens, respectively.
        elif 'pos_node_ids' in key:
            return self.x_programl.size(0)
        elif 'pos_chunk_ids' in key:
            return self.x_src_code.size(0)
        elif 'pos_token_global_ids_in_chunks' in key:
            return self.x_src_code.shape[0] * self.x_src_code.shape[1]  # total number of tokens across all chunks
        else:
            return super().__inc__(key, value, *args, **kwargs)


class SharedGNNData(Data):
    def __init__(self, x_programl, x_src_code, edge_index_shared, edge_attr_shared, x_dummy, **kwargs):
        super().__init__()
        self.x_programl = x_programl
        self.x_src_code = x_src_code
        self.edge_index_shared = edge_index_shared
        self.edge_attr_shared = edge_attr_shared
        self.x_dummy = x_dummy
        # assert x_dummy.shape == (x_programl.shape(0) + x_src_code.shape(0), 1) # for `batch` generation
        # self.attention_mask = attention_mask
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_shared':
            return self.x_programl.size(0) + self.x_src_code.size(0)
        # elif key == 'edge_attr_shared': # be careful! only edge indices should be incremented!
        #     return self.x_programl.size(0) + self.x_src_code.size(0)
        # elif key == 'attention_mask':
        #     return self.x_src_code.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def init_preprocessors_multi_modality():
    # assert FLAGS.encoder_path == None, 'Does not support loading encoders for multi modality data yet'
    preprocessors_programl = init_preprocessors_programl()
    preprocessors_src_code = init_preprocessors_src_code()
    return {'preprocessors_programl': preprocessors_programl, 'preprocessors_src_code': preprocessors_src_code}


def read_multi_modality_data(gexf_file):
    g_programl = read_programl_graph(gexf_file)
    g_src_code = read_source_code(gexf_file)
    return MultiModalityData(g_programl, g_src_code)


def encode_feat_dict_multi_modality(mmo, preprocessors, point=None):
    assert FLAGS.sequence_modeling and FLAGS.data_repr == 'penhance'

    preprocessors_programl = preprocessors['preprocessors_programl']

    n_dict_programl, e_dict_programl = encode_feat_dict_programl(mmo.g_programl, preprocessors_programl, point=point)

    preprocessors_src_code = preprocessors['preprocessors_src_code']

    pc_links = None
    if FLAGS.pc_links:
        pc_links = OrderedDict()

        for nid, (node, ndata) in enumerate(sorted(mmo.g_programl.nodes(data=True))):
            # print(node['type'], type(node['type']))

            # if 'full_text' in ndata and 'pragma' in ndata['full_text']:
            #     print()

            assert nid == node
            line_num = ndata.get('line')
            if line_num is not None:
                assert line_num != 0, 'I assume the line number if 1-indexed'
                line_num -= 1  # turn into 0-indexed
                col_index = ndata.get('col')  # TODO: 0 or 1 indexed?
                col_index -= 1  # turn into 0-indexed
                assert col_index is not None
                if line_num is not None and FLAGS.pc_links_aug != 'grease':
                    pc_links[nid] = {'line': line_num, 'col': col_index, 'ndata': ndata}

        # pc_links[0] = {'line': 0, 'col': 5}
        # pc_links[5] = {'line': 1, 'col': 7}
        # pc_links[9] = {'line': 1, 'col': 0}
        # pc_links[20] = {'line': 1, 'col': 7}
        # pc_links[21] = {'line': 3, 'col': 0}
        # pc_links[22] = {'line': 3, 'col': 0}
        # pc_links[23] = {'line': 3, 'col': 0}
        # pc_links[30] = {'line': 3, 'col': 1}
        # pc_links[30] = {'line': 3, 'col': 5}
        # pc_links[30] = {'line': 3, 'col': 1}

    mmo.pc_links = pc_links

    n_dict_src_code, e_dict_src_code = encode_feat_dict_src_code(mmo.g_src_code, preprocessors_src_code, point=point,
                                                                 pc_links=mmo.pc_links,
                                                                 g_programl=mmo.g_programl,
                                                                 n_dict_programl=n_dict_programl)  # mmo.pc_links would be modified too

    return {'n_dict_programl': n_dict_programl, 'n_dict_src_code': n_dict_src_code}, \
           {'e_dict_programl': e_dict_programl, 'e_dict_src_code': e_dict_src_code}


def fit_preprocessors_multi_modality(preprocessors):
    fit_preprocessors_programl(preprocessors['preprocessors_programl'])
    fit_preprocessors_src_code(preprocessors['preprocessors_src_code'])


PC_LINKS_DICT = {}


def encode_X_torch_multi_modality(mmo, d_node, d_edge, preprocessors, gname, vname):
    dict_programl = encode_X_torch_programl(mmo.g_programl, d_node['n_dict_programl'], d_edge['e_dict_programl'],
                                            preprocessors['preprocessors_programl'], gname, vname, return_Data=False)
    dict_src_code = encode_X_torch_src_code(d_node['n_dict_src_code'], d_edge['e_dict_src_code'],
                                            preprocessors['preprocessors_src_code'], gname, vname, return_Data=False)
    edge_index_programl = dict_programl['edge_index']
    edge_index_src_code = dict_src_code['edge_index']

    if FLAGS.combine_fashion == 'share_final_MLPs':

        if FLAGS.task == 'regression':
            data = SeparateGNNData(
                gname=gname,
                key=vname,

                x_programl=dict_programl['X_node'],
                x_src_code=dict_src_code['X_node'],

                edge_index_programl=edge_index_programl,
                edge_index_src_code=edge_index_src_code,

                edge_attr_programl=dict_programl['X_edge'],

                attention_mask=dict_src_code['attention_mask'],

                pragmas=d_node['pragmas'],
                perf=d_node['perf'],
                actual_perf=d_node['actual_perf'],
                kernel_speedup=d_node['kernel_speedup'],  # base is different per kernel
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
            data = SeparateGNNData(
                gname=gname,
                key=vname,

                x_programl=dict_programl['X_node'],
                x_src_code=dict_src_code['X_node'],

                edge_index_programl=edge_index_programl,
                edge_index_src_code=edge_index_src_code,

                edge_attr_programl=dict_programl['X_edge'],

                attention_mask=dict_src_code['attention_mask'],

                pragmas=d_node['pragmas'],
                perf=d_node['perf'],
            )
        else:
            raise NotImplementedError()

        if FLAGS.feed_p_to_tf:
            data.num_chunks = dict_src_code['X_node'].shape[0]

        if FLAGS.pc_links:
            pos_node_ids = []
            pos_chunk_ids = []
            pos_token_ids = []
            pos_token_global_ids_in_chunks = []
            pos_token_sw_type_in_chunks = []
            for node, d in sorted(mmo.pc_links.items()):
                assert len(d['chunk_ids']) == len(d['token_ids_local']) == len(d['token_ids_global_in_chunks']) > 0
                for i, (chunk_id, token_id_local, token_global_id_in_chunks) in enumerate(
                        zip(d['chunk_ids'], d['token_ids_local'],
                            d['token_ids_global_in_chunks'])):
                    pos_node_ids.append(node)
                    pos_chunk_ids.append(chunk_id)
                    pos_token_ids.append(token_id_local)
                    pos_token_global_ids_in_chunks.append(token_global_id_in_chunks)
                    if FLAGS.pc_links_aug == 'all_line_sw':
                        sw_type = d['strong_weak_types_in_chunks'][i]
                        if sw_type == 'w':
                            to_appennd = [1.0, 0.0]
                        elif sw_type == 's':
                            to_appennd = [0.0, 1.0]
                        else:
                            assert False
                        pos_token_sw_type_in_chunks.append(to_appennd)
                    elif FLAGS.pc_links_aug == 'all_line_swp':
                        sw_type = d['strong_weak_types_in_chunks'][i]
                        if sw_type == 'w':
                            to_appennd = [1.0, 0.0, 0.0]
                        elif sw_type == 's':
                            to_appennd = [0.0, 1.0, 0.0]
                        elif sw_type == 'p':
                            to_appennd = [0.0, 0.0, 1.0]
                        else:
                            assert False
                        pos_token_sw_type_in_chunks.append(to_appennd)
                    elif FLAGS.pc_links_aug == 'all_line_swp_grease':
                        sw_type = d['strong_weak_types_in_chunks'][i]
                        if sw_type == 'w':
                            to_appennd = [1.0, 0.0, 0.0, 0.0]
                        elif sw_type == 's':
                            to_appennd = [0.0, 1.0, 0.0, 0.0]
                        elif sw_type == 'p':
                            to_appennd = [0.0, 0.0, 1.0, 0.0]
                        elif sw_type == 'grease':
                            to_appennd = [0.0, 0.0, 0.0, 1.0]
                        else:
                            assert False
                        pos_token_sw_type_in_chunks.append(to_appennd)
                    elif FLAGS.pc_links_aug == 's_grease':
                        sw_type = d['strong_weak_types_in_chunks'][i]
                        if sw_type == 's':
                            to_appennd = [1.0, 0.0]
                        elif sw_type == 'grease':
                            to_appennd = [0.0, 1.0]
                        else:
                            assert False
                        pos_token_sw_type_in_chunks.append(to_appennd)
                    saver.log_info_at_most(
                        f'{gname}: Node {node} ({d["ndata"]}) matched to token(s) {d["tokens"]} id={token_global_id_in_chunks} in chunk {chunk_id} (line={d["line"]} col={d["col"]})',
                        'pc_link',
                        100)
            data.pos_node_ids = torch.tensor(pos_node_ids)
            data.pos_chunk_ids = torch.tensor(pos_chunk_ids)
            data.pos_token_ids = torch.tensor(pos_token_ids)
            if FLAGS.pc_links_aug in ['all_line_sw', 'all_line_swp', 'all_line_swp_grease', 's_grease']:
                data.pos_token_sw_type_in_chunks = torch.tensor(pos_token_sw_type_in_chunks)
            if FLAGS.node_token_interaction:
                data.pos_token_global_ids_in_chunks = torch.tensor(pos_token_global_ids_in_chunks)

            if FLAGS.pc_links_holdout_ratio > 0:

                psuedo_nids = []
                if FLAGS.pc_links_aug in ['all_line_swp_grease', 's_grease']:
                    for nid, x in enumerate(data.pos_token_sw_type_in_chunks):
                        if FLAGS.pc_links_aug == 'all_line_swp_grease':
                            check_pos = 3
                        elif FLAGS.pc_links_aug == 's_grease':
                            check_pos = 1
                        else:
                            assert False
                        if x[check_pos] == 1: # x[3] corresponds to the last column indicating p node or not
                            psuedo_nids.append(nid)

                # shuffle and then split
                assert len(pos_node_ids) == len(pos_chunk_ids) == len(pos_token_ids)
                perm_indices = torch.randperm(len(pos_node_ids))
                pos_node_ids = data.pos_node_ids[perm_indices]
                pos_chunk_ids = data.pos_chunk_ids[perm_indices]
                pos_token_ids = data.pos_token_ids[perm_indices]

                sp_point = int(len(pos_node_ids) * (1 - FLAGS.pc_links_holdout_ratio))

                data.pos_node_ids_train = pos_node_ids[0:sp_point]
                data.pos_chunk_ids_train = pos_chunk_ids[0:sp_point]
                data.pos_token_ids_train = pos_token_ids[0:sp_point]

                if psuedo_nids:
                    data.pos_node_ids_train = torch.cat((data.pos_node_ids_train, data.pos_node_ids[psuedo_nids]))
                    data.pos_chunk_ids_train = torch.cat((data.pos_chunk_ids_train, data.pos_chunk_ids[psuedo_nids]))
                    data.pos_token_ids_train = torch.cat((data.pos_token_ids_train, data.pos_token_ids[psuedo_nids]))

                data.pos_node_ids_test = pos_node_ids[sp_point:]
                data.pos_chunk_ids_test = pos_chunk_ids[sp_point:]
                data.pos_token_ids_test = pos_token_ids[sp_point:]

                if FLAGS.pc_links_aug in ['all_line_sw', 'all_line_swp', 'all_line_swp_grease', 's_grease']:
                    pos_token_sw_type_in_chunks = data.pos_token_sw_type_in_chunks
                    data.pos_token_sw_type_in_chunks_train = pos_token_sw_type_in_chunks[0:sp_point]
                    data.pos_token_sw_type_in_chunks_test = pos_token_sw_type_in_chunks[sp_point:]

                    if psuedo_nids:
                        data.pos_token_sw_type_in_chunks_train = torch.cat(
                            (data.pos_token_sw_type_in_chunks_train, data.pos_token_sw_type_in_chunks[psuedo_nids]))

                    assert len(data.pos_token_sw_type_in_chunks_train) == len(data.pos_node_ids_train) == len(
                        data.pos_chunk_ids_train) == len(data.pos_token_ids_train)

                if FLAGS.node_token_interaction:
                    pos_token_global_ids_in_chunks = data.pos_token_global_ids_in_chunks[perm_indices]
                    data.pos_token_global_ids_in_chunks_train = pos_token_global_ids_in_chunks[0:sp_point]
                    data.pos_token_global_ids_in_chunks_test = pos_token_global_ids_in_chunks[sp_point:]

                    if psuedo_nids:
                        data.pos_token_global_ids_in_chunks_train = torch.cat(
                            (data.pos_token_global_ids_in_chunks_train,
                             data.pos_token_global_ids_in_chunks[psuedo_nids]))

                    assert len(data.pos_token_global_ids_in_chunks_train) == len(data.pos_node_ids_train) == len(
                        data.pos_chunk_ids_train) == len(data.pos_token_ids_train)

            global PC_LINKS_DICT
            if gname not in PC_LINKS_DICT:
                PC_LINKS_DICT[gname] = mmo.pc_links



    elif FLAGS.combine_fashion == 'share_GNNs_MLPs':

        assert edge_index_programl.shape[0] == edge_index_src_code.shape[0] == 2

        # ordering is important: first programl then src_code
        edge_index_src_code_inc = edge_index_src_code + dict_programl['X_node'].shape[0]
        edge_index_shared = torch.cat((edge_index_programl, edge_index_src_code_inc), dim=1)

        edge_attr_programl = dict_programl['X_edge']
        assert edge_attr_programl.shape[1] == 7  # 7 is the # edge features
        assert edge_attr_programl.shape[0] == edge_index_programl.shape[1]
        num_edges_programl = edge_attr_programl.shape[0]
        num_edges_src_code = edge_index_src_code.shape[1]
        num_edges_total = num_edges_programl + num_edges_src_code
        assert num_edges_total == edge_index_shared.shape[1]
        edge_attr_shared = torch.zeros((num_edges_total, 8))
        edge_attr_shared[0:num_edges_programl, 0:7] = edge_attr_programl
        edge_attr_shared[num_edges_programl:, -1] = 1  # set last column to 1 for the last edges

        num_nodes_total = dict_programl['X_node'].shape[0] + dict_src_code['X_node'].shape[0]
        x_dummy = torch.ones((num_nodes_total, 1))

        if FLAGS.add_pragma_links:
            pnode_ptext_dict_programl = dict_programl['d_node']['X_pragma_nodes']
            pnode_ptext_dict_src_code = dict_src_code['d_node']['X_pragma_nodes']
            for node, ptext in pnode_ptext_dict_src_code.items():
                ptext_to_verify = ''.join(dict_src_code['d_node']['g_new'].nodes(data=True)[node]['tokens'])
                assert ptext in ptext_to_verify
            links = {}
            rev = {v: k for k, v in pnode_ptext_dict_src_code.items()}
            assert len(pnode_ptext_dict_programl) == len(pnode_ptext_dict_src_code) == len(rev)
            edge_index_plinks = []
            for node1, ptext1 in pnode_ptext_dict_programl.items():
                node2 = rev[ptext1] + dict_programl['X_node'].shape[0]
                links[node1] = node2
                edge_index_plinks.append((node1, node2))
                edge_index_plinks.append((node2, node1))
            edge_index_plinks = torch.LongTensor(edge_index_plinks).t()
            edge_index_shared = torch.cat((edge_index_shared, edge_index_plinks), dim=1)
            num_edges_total += 2 * len(links)
            assert edge_index_shared.shape == (2, num_edges_total)
            edge_attr_shared_new = torch.zeros((num_edges_total, 9))
            edge_attr_shared_new[0:edge_attr_shared.shape[0], 0:8] = edge_attr_shared
            edge_attr_shared_new[edge_attr_shared.shape[0]:, -1] = 1  # set last column to 1 for the last edges
            edge_attr_shared = edge_attr_shared_new
            assert edge_attr_shared.shape == (num_edges_total, 9)

        data = SharedGNNData(
            gname=gname,
            key=vname,

            x_programl=dict_programl['X_node'],
            x_src_code=dict_src_code['X_node'],

            edge_index_shared=edge_index_shared,

            edge_attr_shared=edge_attr_shared,

            x_dummy=x_dummy,

            attention_mask=dict_src_code['attention_mask'],

            pragmas=d_node['pragmas'],
            perf=d_node['perf'],
            actual_perf=d_node['actual_perf'],
            kernel_speedup=d_node['kernel_speedup'],  # base is different per kernel
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
    else:
        raise NotImplementedError()

    return data
