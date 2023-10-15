'''
handles sequence and AST representations for source code, which is different from the programl representation at the assembly language level.
'''

from rose_gen_dot import get_ast_graph
from saver import saver
from utils import get_root_path, create_edge_index, NON_OPT_PRAGMAS, WITH_VAR_PRAGMAS, check_any_in_str, coo_to_sparse, \
    load
from config import FLAGS
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, CodeGenModel
# from transformers import T5ForConditionalGeneration
from our_codet5 import T5ForConditionalGeneration
import torch
from os.path import join, basename
from glob import glob
from pathlib import Path
import networkx as nx
import numpy as np

from collections import defaultdict, Counter, OrderedDict

tokenizer = None


def create_tokenizer():
    new_words = []
    if FLAGS.code_encoder == 'codebert':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        MAX_LEN = 512
    elif FLAGS.code_encoder == 'graphcodebert':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        MAX_LEN = 512
    elif FLAGS.code_encoder == 'codet5':
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        MAX_LEN = 256  # can be larger, actually
    elif FLAGS.code_encoder == 'codet5-large':
        tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-large')
        MAX_LEN = 256  # can be larger, actually
    elif 'codegen' in FLAGS.code_encoder:
        tokenizer = AutoTokenizer.from_pretrained(f'Salesforce/{FLAGS.code_encoder}')
        MAX_LEN = 2048
        # xxx = tokenizer.cls_token
        tokenizer.cls_token = '<|startoftext|>'  # for some reason it is missing in this tokenizer
        new_words += [tokenizer.cls_token]
        tokenizer.pad_token = '<pad>'
        new_words += [tokenizer.pad_token]
        tokenizer.mask_token = '<mask>'
        new_words += [tokenizer.mask_token]
        # xxx = tokenizer.cls_token
        # print()
    else:
        raise NotImplementedError()
    if hasattr(FLAGS, 'max_code_tokens_len') and FLAGS.max_code_tokens_len is not None:
        saver.log_info(
            f'tokenizer: MAX_LEN set to FLAGS.max_code_tokens_len={FLAGS.max_code_tokens_len} instead of default {MAX_LEN}')
        # assert 3 <= FLAGS.max_code_tokens_len <= MAX_LEN
        MAX_LEN = FLAGS.max_code_tokens_len  # bound the max length to save memory
        assert MAX_LEN >= 1
    if FLAGS.preserve_keywords or len(new_words) > 0:
        vocab = tokenizer.get_vocab()
        len_before = len(vocab)
        saver.log_info(f'Before: Tokenizer has {len(vocab)} tokens in its vocabulary')
        new_words += \
            ['pragma', 'ACCEL',
             'PIPELINE', 'TILE', 'PARALLEL',
             'reduction', 'auto', 'FACTOR',
             '__PIPE__', '__TILE__', '__PARA__', ] + \
            NON_OPT_PRAGMAS + WITH_VAR_PRAGMAS  # what if a new functiona name? should we preserve all function names? TODO: resolve this concern -- maybe a good furue direction
        assert FLAGS.pk_version >= 1
        if FLAGS.pk_version != 1:
            assert FLAGS.pk_version == 2
            # Note: FUNCTION_NAMES1 and VARIABLE_NAMES1 are extracted by running v2_db=True on 03/17/2023.
            #       FUNCTION_NAMES2 and VARIABLE_NAMES2 are extracted by running v2_db=False on 03/17/2023.
            FUNCTION_NAMES1 = {'stencil3d', 'kernel_bicg', 'aes_mixColumns_1', 'rj_xtime_1', 'kernel_doitgen',
                               'kernel_gesummv', 'kernel_syrk', 'kernel_trmm', 'kernel_2mm', 'kernel_gemver', 'ellpack',
                               'kernel_gemm', 'kernel_symm', 'aes_addRoundKey_1', 'aes256_encrypt_ecb',
                               'kernel_covariance', 'spmv', 'aes_subBytes_1', 'kernel_fdtd_2d', 'aes_addRoundKey_cpy_1',
                               'gemm', 'aes_shiftRows_1', 'kernel_atax', 'bbgemm', 'stencil', 'needwun',
                               'aes_expandEncKey_1', 'kernel_mvt'}
            VARIABLE_NAMES1 = {'ex', 'm', 'out', 'sbox[256]', 'w', 'vec', 'key', 'b', 'k_col', 'SEQB', 'i_row', 'mul',
                               'hz', 'Si', 'B', 'y', 'i', 't', 'v1', 'row', 'float_n', 'nk', 'temp2', 'tmp_end', 'ko',
                               'rowDelimiters', 'mul0', 'C1', 'i_col', 'row_up', 'mul1', 'ptr', 'C4', 'SEQA', 'int',
                               'sum0', 'filter', 'register', 'char', 'temp', '_s_i_0', 's', 'sum_tmp', 'e', 'long',
                               'orig', 'double', 'n', 'alpha', 'buf', 'cov', '_in_kk', 'sum', 'nzval', 'm1', 'kk',
                               'b_idx', 'prod', 'k1', 'r', 'M', 'mean', 'cpk', 'left', 'data', 'm2', 'c', '_in_s_i',
                               'cols', 'alignedB', 'tmp_begin', 'nj', 'a_idx', '_in_ko', 'jj', 'x', 'd', 'mult', 'D',
                               'j', 'up', 'up_left', 'sum1', 'a', 'u1', 'np', 'max', 'score', 'tmp', 'unsigned', 'x2',
                               'y_2', 'p', '_s_i', 'rcon', 'temp_x', 'q', 'C', 'v2', 'k_row', 'alignedA', 'val', 'ctx',
                               'A', 'const', 'C0', 'nx', 'u2', '_fict_', 'ey', 'ni', 'nq', 'a_str_idx', 'tmax', 'ny',
                               'nl', 'beta', 'z', 'k', 'sol', 'nr', 'x1', '_in_jj', 'k2', 'b_str_idx', 'rc', 'y_1'}
            FUNCTION_NAMES2 = {'aes256_encrypt_ecb', 'aes_shiftRows_1', 'kernel_trmm', 'gemm', 'kernel_doitgen', 'spmv',
                               'kernel_2mm', 'kernel_heat_3d', 'kernel_gesummv', 'needwun', 'kernel_fdtd_2d',
                               'aes_subBytes_1', 'kernel_gemm', 'rj_xtime_1', 'aes_addRoundKey_1', 'kernel_atax',
                               'aes_expandEncKey_1', 'stencil3d', 'kernel_jacobi_1d', 'kernel_bicg', 'ellpack',
                               'aes_mixColumns_1', 'aes_addRoundKey_cpy_1', 'kernel_jacobi_2d', 'bbgemm', 'stencil',
                               'kernel_seidel_2d', 'kernel_symm', 'md_kernel', 'kernel_mvt', 'kernel_syrk',
                               'kernel_adi',
                               'kernel_gemver', 'kernel_3mm'}
            VARIABLE_NAMES2 = {'k_row', 'score', 'unsigned', 'j', 'left', 'temp2', 'jj', 'u', 'beta', 'tmp', 'i_y',
                               'SEQA', 'vec', 'ptr', 'nx', 'k_col', '_in_kk', '_in_ko', 'alpha', 'potential', 'temp_x',
                               'u1', 'B1', 'G', 'C4', 'mul1', 'cpk', 'max', 'D', 'mul2', 'y', 'NL', 'B', 'u2', '_in_jj',
                               'k1', 'position_y', '_in_j_0', 'j_y', 'e', 'mul', '_s_i_0', 'nj', 'i_x', 'delx', 'x1',
                               'a_str_idx', 'sbox[256]', 'row', 'prod', 'i_row', 'ni', 'c', 'nm', 'char', 'A', 'tmax',
                               'position_x', 'force', 'fx', 'kk', 'tmp_begin', 'int', 'up_left', 'fy', 'sum', 'nq',
                               'ko',
                               'v2', 'nr', 'delz', 'alignedA', 'double', 'rowDelimiters', 'fz', 'r2inv', 'buf', 'mul0',
                               '_in_j', 'rcon', 'sum1', 't', 'i', 'force_y', 'r6inv', 'tsteps', 'position_z', 'dely',
                               'np', 'w', 'filter', 'F', 'const', 'v', 'out', 'ny', 'M', 'DX', 'DY', 'nl', 'key', 'd',
                               'C', 'm1', 'orig', 'n', 'tmp_end', 'ey', 'register', 'SEQB', 'temp', 'C0', 'j_x', 'b',
                               'v1', 'sum_tmp', '_fict_', 'a', 'C1', 'Si', 'p', 'force_x', 'i_col', 'y_1', 'a_idx', 'E',
                               '_in_s_i', 'DT', 's', 'cols', 'm2', 'sum0', 'ctx', 'B2', 'hz', 'force_z', 'up', 'b_idx',
                               'val', 'z', 'mult', 'y_2', 'long', 'k2', 'nk', '_s_i', 'j_z', 'sol', 'm', 'jidx', 'k',
                               'i_z', 'alignedB', 'row_up', 'ex', 'rc', 'x', 'r', 'f', 'q', 'x2', 'nzval', 'b_str_idx'}
            new_words += FUNCTION_NAMES1
            new_words += VARIABLE_NAMES1
            new_words += FUNCTION_NAMES2
            new_words += VARIABLE_NAMES2
            new_words += ['TILING']
        if FLAGS.pk_version != 1:
            new_words = sorted(list(set(new_words)))
        saver.log_info(f'new_workds={new_words} (len(new_words)={len(new_words)})')
        for word in new_words:
            if word not in vocab.keys():
                tokenizer.add_tokens(word)
            else:
                pass
                # saver.log_info(f'{word} already in tokenizer\'s vocabulary')
        saver.log_info(
            f'After: Tokenizer has {len(tokenizer)} tokens in its vocabulary ({len(tokenizer) - len_before} added)')

    return tokenizer, MAX_LEN


def _get_src_files():
    if FLAGS.check_release_db:
        rtn = glob(join(get_root_path(), FLAGS.dse_database_name, '**', '*.c'), recursive=True)
    else:
        rtn = glob(join(get_root_path(), FLAGS.dse_database_name, 'programl', '**', '*.c'), recursive=True)
        if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
            rtn = glob(join(get_root_path(), FLAGS.dse_database_name, 'multi_modality', '**', '*.c'), recursive=True)
    return rtn


SOURCE_FILES = None
if FLAGS.sequence_modeling:  # and FLAGS.force_regen:
    SOURCE_FILES = _get_src_files()
    tokenizer, MAX_LEN = create_tokenizer()  # TODO: this code seems fine but when loading a trained model, its flags will be replaced in main(), but this code is executed before flags are replaced... so need to refactor it by wrapping it into a function and call it in appropriate places
    # xxx = create_code_encoder()
    # print()


def update_tokenizer():
    if FLAGS.sequence_modeling:
        global SOURCE_FILES, tokenizer, MAX_LEN
        SOURCE_FILES = _get_src_files()
        tokenizer, MAX_LEN = create_tokenizer()


def create_code_encoder():
    if FLAGS.code_encoder == 'codebert':
        encoder = AutoModel.from_pretrained("microsoft/codebert-base")
    elif FLAGS.code_encoder == 'graphcodebert':
        encoder = AutoModel.from_pretrained("microsoft/graphcodebert-base")
    elif FLAGS.code_encoder == 'codet5':
        encoder = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small').encoder
    elif FLAGS.code_encoder == 'codet5-large':
        encoder = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-large').encoder
    elif 'codegen' in FLAGS.code_encoder:
        # encoder = T5ForConditionalGeneration.from_pretrained(f'Salesforce/{FLAGS.code_encoder}').encoder
        checkpoint = f"Salesforce/{FLAGS.code_encoder}"
        encoder = CodeGenModel.from_pretrained(checkpoint)
    else:
        raise NotImplementedError()

    if hasattr(FLAGS, 'replace_with_random_weights') and FLAGS.replace_with_random_weights:
        assert FLAGS.subtask != 'inference'
        new_params = {}
        for name, param in encoder.named_parameters():
            new_params[f'{name}'] = torch.nn.Parameter(torch.randn(param.shape))
        encoder.load_state_dict(new_params)
        saver.log_info(f'Critical!!!!! Replaced encoder with {len(new_params)} random parameters')

    if FLAGS.preserve_keywords:
        def _get_encoder_vocab_size(encoder):
            if hasattr(encoder, 'embed_tokens'):
                return str(encoder.embed_tokens)
            elif hasattr(encoder, 'vocab_size'):
                return encoder.vocab_size
            elif hasattr(encoder, 'embeddings'):
                return str(encoder.embeddings)
            elif hasattr(encoder, 'transformer'):
                return str(encoder.transformer.vocab_size)
            else:
                raise NotImplementedError()

        saver.log_info(f'encoder.resize_token_embeddings')
        saver.log_info(f'Before: {_get_encoder_vocab_size(encoder)}')
        global tokenizer
        if tokenizer is None:
            tokenizer, _ = create_tokenizer()
        encoder.resize_token_embeddings(len(tokenizer))
        saver.log_info(f'After: {_get_encoder_vocab_size(encoder)}')

    return encoder, get_code_encoder_dim()


def get_code_encoder_dim():
    if FLAGS.code_encoder == 'codebert':
        DIM = 768
    elif FLAGS.code_encoder == 'graphcodebert':
        DIM = 768
    elif FLAGS.code_encoder == 'codet5':
        DIM = 512
    elif FLAGS.code_encoder == 'codet5-large':
        DIM = 1024
    elif FLAGS.code_encoder == 'codegen-350M-multi':
        DIM = 1024
    else:
        raise NotImplementedError()
    return DIM


def get_code_encoder_num_layers():
    if FLAGS.code_encoder == 'codebert':
        raise NotImplementedError()
    elif FLAGS.code_encoder == 'graphcodebert':
        num_layers = 12
    elif FLAGS.code_encoder == 'codet5':
        num_layers = 6
    elif FLAGS.code_encoder == 'codet5-large':
        raise NotImplementedError()
    elif FLAGS.code_encoder == 'codegen-350M-multi':
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    return num_layers


def init_preprocessors_src_code():
    ntypes = Counter()

    # if FLAGS.encoder_path != None:
    #     encoders = load(FLAGS.encoder_path)
    #     enc_ntype = encoders['enc_ntype']
    #     enc_ftype_edge = encoders['enc_ftype_edge']
    #     enc_ptype_edge = encoders['enc_ptype_edge']
    # else:
    ## handle_unknown='ignore' is crucial for handling unknown variables of new kernels
    enc_ntype = OneHotEncoder(handle_unknown='ignore')
    enc_ptype = OneHotEncoder(handle_unknown='ignore')
    enc_itype = OneHotEncoder(handle_unknown='ignore')
    enc_ftype = OneHotEncoder(handle_unknown='ignore')
    enc_btype = OneHotEncoder(handle_unknown='ignore')

    enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
    enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')

    X_ntype_all = []
    X_ptype_all = []
    X_itype_all = []
    X_ftype_all = []
    X_btype_all = []

    edge_ftype_all = []
    edge_ptype_all = []

    return {'counters': {'ntypes': ntypes, 'num_tokens': [], 'num_matched_pairs': []},
            'encoders': {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype, 'enc_itype': enc_itype, 'enc_ftype': enc_ftype,
                         'enc_btype': enc_btype, 'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge},
            'X_all': {'X_ntype_all': X_ntype_all, 'X_ptype_all': X_ptype_all, 'X_itype_all': X_itype_all,
                      'X_ftype_all': X_ftype_all, 'X_btype_all': X_btype_all,
                      'edge_ftype_all': edge_ftype_all, 'edge_ptype_all': edge_ptype_all}}


def read_source_code(gexf_file):
    n = basename(gexf_file).split('_')[0]
    if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
        n = basename(gexf_file).split('.')[0]
    if not FLAGS.check_release_db:
        if basename(gexf_file) == 'stencil_stencil2d_processed_result.gexf':
            n = 'stencil_stencil2d'
        if basename(gexf_file) == 'stencil_stencil2d.gexf':
            n = 'stencil_stencil2d'
    gname = n

    found_file = None
    if SOURCE_FILES is None:
        raise RuntimeError(
            f'SOURCE_FILES is None -- check if config.py is correct (maybe load a wrong model?) SOURCE_FILES')
    for source_file in SOURCE_FILES:
        if FLAGS.check_release_db:
            c = basename(source_file).split('.')[0].split('_kernel')[0]
        else:
            c = basename(source_file).split('.')[0]
        if gname == c:
            if found_file is not None:
                raise ValueError(f'gname={gname} source_file={source_file} found_file={found_file}')
            found_file = source_file
    if found_file is None:
        raise ValueError(f'Cannot find {gname}\'s source code file in {len(SOURCE_FILES)} source files: {SOURCE_FILES}')
    saver.log_info(f'Found source code file for {gname}: {found_file}')
    text = Path(found_file).read_text()
    if FLAGS.data_repr == 'ast':
        g = get_ast_graph(text, found_file, gname)
    else:
        g = nx.Graph()
        g.add_node(0, text=text)
        g.gname = gname
    return g


def encode_feat_dict_src_code(g, preprocessors, point, pc_links=None, g_programl=None, n_dict_programl=None):
    if pc_links is not None:
        assert FLAGS.data_repr != 'ast', 'Not implemented yet for AST'
    if FLAGS.data_repr == 'ast':
        return _encode_feat_dict_sequence_AST(g, preprocessors, point)
    else:
        return _encode_feat_dict_sequence_pure_seq(g, preprocessors, point, pc_links, g_programl,
                                                   n_dict_programl=n_dict_programl)


CODE_LEN_LI = []
FUNCTION_NAMES = set()
VARIABLE_NAMES = set()
ALREADY_PRINTED_GNAMES_NODES = set()


def _encode_feat_dict_sequence_AST(g, preprocessors, point):
    ntypes = preprocessors['counters']['ntypes']
    global CODE_LEN_LI, FUNCTION_NAMES_LI, VARIABLE_NAMES, ALREADY_PRINTED_GNAMES_NODES

    def _tokenizer_short_code(code_text, nid):
        tokens = tokenizer.tokenize(code_text)
        if not 0 <= len(tokens) <= MAX_LEN - 2:
            tokens_take = tokens[0:MAX_LEN - 2]  # TODO: handle long text
            if (g.gname, nid) not in ALREADY_PRINTED_GNAMES_NODES:
                saver.log_info(
                    f'Assume each AST node has a short code fragment (<{MAX_LEN} tokens); nid={nid}; len(tokens)={len(tokens)}; code_text={code_text}\ntokens_take={tokens_take}')
                ALREADY_PRINTED_GNAMES_NODES.add((g.gname, nid))
            tokens = tokens_take
        CODE_LEN_LI.append(len(tokens))

        rtn = [tokenizer.cls_token] + tokens + [tokenizer.eos_token]
        # if len(rtn) > MAX_LEN:
        #     raise ValueError(f'Check in _tokenizer_short_code: len(tokens)={len(rtn)} > MAX_LEN={MAX_LEN}; tokens={rtn}')
        return rtn

    def _find_block_node(g, nid):
        block_nodes = []
        for neighbor in g.neighbors(nid):
            if g.nodes[neighbor]['type'] == 'SgBasicBlock':
                block_nodes.append(neighbor)
        if len(block_nodes) != 1:
            raise ValueError(f'AST: {len(block_nodes)} block nodes for node {nid}! Assume must be 1')
        return block_nodes[0]

    g_new = g.copy()
    X_ntype = []  # node type <attribute id="3" title="type" type="long" />
    X_contextnids = []  # 0 or 1 showing context node
    X_pragmanids = []  # 0 or 1 showing pragma node
    X_pragma_dict_repr = {**point}
    X_pragma_trans = defaultdict(list)

    for nid, (node, ndata) in enumerate(sorted(g_new.nodes(data=True))):  # tricky; be consistent in node ordering
        # print(node['type'], type(node['type']))
        assert nid == node
        if ntypes is not None:
            ntypes[ndata['type']] += 1

            if ndata['type'] == 'SgFunctionParameterList':
                FUNCTION_NAMES.add(ndata['content'].split('(')[0].split()[-1])
            elif ndata['type'] == 'SgInitializedName':
                for x in ndata['content'].split():
                    VARIABLE_NAMES.add(x)

            if ndata['type'] == 'SgPragmaDeclaration':
                if ndata['content'] == '':
                    raise ValueError(f'Some error in AST! Perhaps messed up dot file by rose compiler! Debug it!')

        if 'pragma' in ndata['content']:
            # print(ndata['content'])
            code_text = ndata['content']
            assert ndata['type'] == 'SgPragmaDeclaration'
            assert code_text[0:8] == '#pragma '

            sorted_point = sorted(list(point.items()), key=lambda key: len(key[0]))
            for key, value in sorted_point:
                if value == '':
                    value = 'empty'  # TODO: "special" token
                c = code_text.count(key)
                if c == 0:
                    continue
                elif c == 1:
                    new_text_repr = f'{key}={value}'
                    code_text = code_text.replace(key, new_text_repr)
                    nx.set_node_attributes(g_new, {nid: code_text}, name='content')  # replace the content
                else:
                    saver.log_info(f'Kernel {g.gname} has {key} appearing more than once ({c} times)')
                    exit()
            X_pragmanids.append(1)
            X_contextnids.append(0)

            if FLAGS.ptrans:
                if '=' in code_text:
                    ptype = code_text.split('=')[0].split('auto')[0]
                    ignore_this_prsgma = False
                else:
                    ptype = None
                    # assert code_text == '#pragma ACCEL kernel'
                    ignore_this_prsgma = True
                if not ignore_this_prsgma:
                    if FLAGS.pragma_scope == 'block':
                        block = _find_block_node(g_new, nid)
                        X_pragma_trans[ptype].append(block)
                        scope = [block]
                    else:
                        raise NotImplementedError()
                    for s in scope:
                        nx.set_node_attributes(g, {s: 1}, name=f'ptrans_{ptype}_scope')

        else:
            X_pragmanids.append(0)
            X_contextnids.append(1)
            code_text = ndata['content']

        X_ntype.append([ndata['type']])

        code_tokens = _tokenizer_short_code(code_text, nid)
        nx.set_node_attributes(g_new, {nid: code_tokens}, name='tokens')  # new attribute

    assert type(g_new) is nx.DiGraph
    node_dict = {'g_new': g_new,
                 'X_ntype': X_ntype,
                 'X_contextnids': torch.FloatTensor(np.array(X_contextnids)),
                 'X_pragmanids': torch.FloatTensor(np.array(X_pragmanids)),
                 'X_pragma_dict_repr': X_pragma_dict_repr,
                 'X_pragma_trans': {}}

    # for node, data in sorted(g_new.nodes(data=True)):
    #     tokens = data['tokens']
    #     if len(tokens) > MAX_LEN:
    #         raise ValueError(f'Check: len(tokens)={len(tokens)} > MAX_LEN={MAX_LEN}; tokens={tokens}')

    if FLAGS.ptrans:
        for tstype, tnids in X_pragma_trans.items():
            node_dict['X_pragma_trans'][tstype] = torch.LongTensor(np.array(tnids))

    # def encode_edge_dict_AST(g):
    X_ftype = []  # flow type <attribute id="5" title="flow" type="long" />
    X_ptype = []  # position type <attribute id="6" title="position" type="long" />

    if FLAGS.sequence_modeling:
        if FLAGS.data_repr != 'ast':
            assert g.number_of_nodes() == 1
    for nid1, nid2, edata in g.edges(
            data=True):  # TODO: node ordering; seems fine due to being consistent with edge index generation
        X_ftype.append([edata['elabel']])  # tricky: call it ftype but in reality just elabel (edge type for AST)
        if FLAGS.ptrans:
            X_ptype.append([edata['direction']])

    edge_dict = {'X_ftype': X_ftype, 'X_ptype': X_ptype}

    preprocessors['X_all']['X_ntype_all'] += node_dict['X_ntype']

    preprocessors['X_all']['edge_ftype_all'] += edge_dict['X_ftype']
    preprocessors['X_all']['edge_ptype_all'] += edge_dict['X_ptype']

    return node_dict, edge_dict


def _encode_feat_dict_sequence_pure_seq(g, preprocessors, point, pc_links, g_programl, n_dict_programl=None):
    num_tokens = preprocessors['counters']['num_tokens']
    num_matched_pairs = preprocessors['counters']['num_matched_pairs']

    assert g.number_of_nodes() == 1
    text = g.nodes[0]['text']
    len_before = len(text)

    num_tokens.append(len(tokenizer.tokenize(text)))

    sorted_point = sorted(list(point.items()), key=lambda key: len(key[0]))

    text_simplified = []
    X_pragma_nodes = {}

    if hasattr(FLAGS, 'pc_links_aug') and FLAGS.pc_links_aug in ['all_line_swp', 'all_line_swp_grease']:
        pc_links = _aug_pc_links_with_pragma_nodes(pc_links, n_dict_programl, text)

    for key, value in sorted_point:
        if value == '':
            value = 'empty'  # TODO: "special" token
        c = text.count(key)
        if c == 0:
            raise ValueError(
                f'The source code text of {g.gname} does not contain point key {key} -- point is {point}; text is {text}')
        if c >= 2:
            saver.log_info(f'Kernel {g.gname} has {key} appearing more than once ({c} times) (okay but double check)')
        # assert c == 1
        new_text_repr = f'{key}={value}'  # '__PARA_L0__' --> __PARA_Lo__=empty
        if FLAGS.data_repr == 'full':
            pass
        elif FLAGS.data_repr in ['simplified', 'penhance', 'ast']:  # TODO remove AST
            text_simplified.append(new_text_repr)
        else:
            assert False
        text = text.replace(key, new_text_repr, 1)  # TODO
    assert len(text) > len_before

    if FLAGS.data_repr in ['full', 'penhance']:
        # text_to_use = g.nodes[0]['text'] # TODO: bug
        text_to_use = text
    elif FLAGS.data_repr == 'simplified':
        text_to_use = ' '.join(text_simplified)
    else:
        assert False

    if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
        # very fancy: tokenizer line by line and handle the programl to source code links
        tokens, pc_links = _tokenize_with_programl_src_code_links(text_to_use, pc_links)
    else:
        tokens = tokenizer.tokenize(text_to_use)

    if pc_links is not None:
        tot_links = 0
        for _, d in pc_links.items():
            tot_links += len(d['token_ids_global'])
        num_matched_pairs.append(tot_links)

    g_new = nx.Graph()
    s = 0
    assert len(tokens) >= 1
    if not (0 <= FLAGS.chunk_offset < MAX_LEN - 2):
        raise ValueError(
            f'FLAGS.chunk_offset={FLAGS.chunk_offset} wrong; must be 0<FLAGS.chunk_offset<=MAX_LEN-2={MAX_LEN - 2}')

    token_id_chunk_token_id_map = defaultdict(list)
    chunk_id = 0
    token_id_global_in_chunks = 0
    cls_token_gids = []
    if hasattr(FLAGS, 'feed_p_to_tf') and FLAGS.feed_p_to_tf and \
            hasattr(FLAGS, 'interleave_GNN_transformer') and FLAGS.interleave_GNN_transformer:
        shift_for_feed_x = True
    else:
        shift_for_feed_x = False
    while True:
        e = s + MAX_LEN - 2
        tokens_take = [tokenizer.cls_token] + tokens[s:e] + [tokenizer.eos_token]
        if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:

            if not shift_for_feed_x:
                cls_token_gids.append(token_id_global_in_chunks)  # cls token

            token_id_local = 1  # critical! cls_token has been appended at the front!
            token_id_global_in_chunks += 1

            if shift_for_feed_x:
                cls_token_gids.append(token_id_global_in_chunks)  # cls token

            # Below code is tricky. When feed programl embedding to transformer,
            # need to shift the token ids by 1 to the right, because
            # in our_codet5.py, this additional "token" is appended to the beginning of EVERY chunk.
            if shift_for_feed_x:
                token_id_local += 1
                token_id_global_in_chunks += 1

            for token_id_global in range(s, e):
                token_id_chunk_token_id_map[token_id_global].append(
                    {'chunk_id': chunk_id, 'token_id_local': token_id_local,
                     'token_id_global_in_chunks': token_id_global_in_chunks})
                token_id_local += 1
                token_id_global_in_chunks += 1

            # TODO: (05/05/2023) this is the bug found! need to include the eos token as well!
            token_id_global_in_chunks += 1
        g_new.add_node(g_new.number_of_nodes(), tokens=tokens_take)
        s = e
        if FLAGS.chunk_offset != 0:
            s = s - FLAGS.chunk_offset  # go back <FLAGS.chunk_offset> number of tokens
        chunk_id += 1
        if s >= len(tokens):
            break

    if FLAGS.data_repr == 'penhance':
        for ptext in text_simplified:
            ptokens = [tokenizer.cls_token] + tokenizer.tokenize(ptext) + [tokenizer.eos_token]
            g_new.add_node(g_new.number_of_nodes(), tokens=ptokens)
            X_pragma_nodes[g_new.number_of_nodes() - 1] = ptext.split('=')[0]

    if hasattr(FLAGS, 'pc_links') and FLAGS.pc_links:
        for node, d in pc_links.items():
            # The below four items will be used by data_multi_modality.py to creare the final torch id tensors.
            d['chunk_ids'] = []
            d['token_ids_local'] = []
            d['token_ids_global_in_chunks'] = []
            d['strong_weak_types_in_chunks'] = []
            for i, token_id_global in enumerate(d['token_ids_global']):
                if FLAGS.pc_links_aug in ['all_line_sw', 'all_line_swp', 'all_line_swp_grease', 's_grease']:
                    assert len(d['token_ids_global']) == len(d['strong_weak_types'])
                    sw_type = d['strong_weak_types'][i]
                for d_chunk_local_id_info in token_id_chunk_token_id_map[token_id_global]:
                    d['chunk_ids'].append(d_chunk_local_id_info['chunk_id'])
                    d['token_ids_local'].append(d_chunk_local_id_info['token_id_local'])
                    d['token_ids_global_in_chunks'].append(d_chunk_local_id_info['token_id_global_in_chunks'])
                    if FLAGS.pc_links_aug in ['all_line_sw', 'all_line_swp', 'all_line_swp_grease', 's_grease']:
                        d['strong_weak_types_in_chunks'].append(sw_type)

        if FLAGS.pc_links_aug in ['all_line_swp_grease', 'grease', 's_grease']:
            # Check psuedo node exists.
            pnode_id = g_programl.number_of_nodes() - 1
            assert type(g_programl) is nx.DiGraph and g_programl.degree[pnode_id] == g_programl.number_of_nodes() * 2
            pc_links[pnode_id] = {'line': -1, 'col': -1, 'ndata': g_programl.nodes.data()[pnode_id]}

            num_chunks = chunk_id
            pc_links[pnode_id]['chunk_ids'] = list(range(num_chunks))
            pc_links[pnode_id]['token_ids_local'] = [0] * num_chunks
            assert len(cls_token_gids) == num_chunks
            pc_links[pnode_id]['token_ids_global_in_chunks'] = cls_token_gids
            pc_links[pnode_id]['strong_weak_types_in_chunks'] = ['grease'] * num_chunks
            pc_links[pnode_id]['tokens'] = ['<cls> (psuedo node)']

    for n in range(g_new.number_of_nodes() - 1):  # TODO: line graph?
        g_new.add_edge(n, n + 1)
    for n in range(g_new.number_of_nodes()):  # TODO: self loop to avoid GNN receiving an empty edge_index?
        g_new.add_edge(n, n)

    if FLAGS.add_edges_fc_graph:
        for n1 in range(g_new.number_of_nodes()):
            for n2 in range(g_new.number_of_nodes()):
                g_new.add_edge(n1, n2)

    # print(f'Created {len(g_new)} nodes with {g_new.number_of_edges()} edges')

    # if len(tokens) > MAX_LEN:
    #     g_new = nx.Graph()
    #
    #     saver.log_info()

    # return {'g_new': g_new, 'X_contextnids': [], 'X_pragmanids': [], 'X_pseudonids': [], 'pragmas': [], 'X_pragma_trans': {}}

    return {'g_new': g_new,
            'X_pragma_trans': {}, 'X_pragma_nodes': X_pragma_nodes}, {}


def _aug_pc_links_with_pragma_nodes(pc_links, n_dict_programl, text):
    all_lines = text.split('\n')
    linetext_to_linenum_dict = {}
    for line_num, line in enumerate(all_lines):
        linetext_to_linenum_dict[line] = line_num
    for nid, ptext in n_dict_programl['X_pragma_ptext'].items():
        line_num = linetext_to_linenum_dict.get(ptext)
        if line_num is None:
            raise ValueError(
                f'Trying to find ptext={ptext} (node {nid}) but it does not appear in the source code text')
        pc_links[nid] = {'line': line_num, 'col': 0, 'ndata': {'is_pragma': True}}
    return pc_links


def _tokenize_with_programl_src_code_links(text_to_use, pc_links):
    assert pc_links is not None
    line_num_to_nodes_lookup = defaultdict(list)
    all_lines = text_to_use.split('\n')
    for node, d in pc_links.items():
        line_num = d['line']
        if not (0 <= line_num < len(all_lines)):
            raise ValueError(f'Wrong line number {line_num} for node {node}: {len(all_lines)} in total')
        line_num_to_nodes_lookup[line_num].append(node)

    all_tokens = []

    for line_num, line in enumerate(all_lines):
        nodes_to_process = line_num_to_nodes_lookup.get(line_num)

        if nodes_to_process is not None:
            assert len(nodes_to_process) > 0
            col_indices = set()
            for node in nodes_to_process:
                line_num_, col_index = pc_links[node]['line'], pc_links[node]['col']
                assert line_num == line_num_
                if not (0 <= col_index < len(line)):
                    raise ValueError(f'Something wrong about the programl to source code link (pc link): '
                                     f'Node {node} says line {line_num} col {col_index} but that line has {len(line)} chars; '
                                     f'Perhaps the source code used to generate the pc link is NOT exactly the same as the '
                                     f'one using here!\nline={line}')
                col_indices.add(col_index)

            if FLAGS.pc_links_aug == 'all_line':
                assert line != ''
                tokens = tokenizer.tokenize(line)
                token_ids = []
                for i in range(len(tokens)):
                    token_ids.append(len(all_tokens) + i)
                all_tokens += tokens
                for node in nodes_to_process:
                    pc_links[node]['token_ids_global'] = token_ids  # all tokens on this line
                    pc_links[node]['tokens'] = tokens
            elif FLAGS.pc_links_aug is None or FLAGS.pc_links_aug in ['all_line_sw', 'all_line_swp',
                                                                      'all_line_swp_grease', 's_grease']:
                partition_indices = sorted(list(col_indices))

                be_indices = []
                begin = 0
                assert len(line) > 0
                for col_index in partition_indices:
                    if begin == col_index:
                        assert begin == col_index == 0  # it would be pretty weird to have s=0, e=0 (empty string)
                    else:
                        be_indices.append((begin, col_index))
                    begin = col_index
                be_indices.append((begin, len(line)))

                tokens_this_line_dict = OrderedDict()
                tokens_this_line = []
                token_ids_this_line = []
                global_to_local_ids_map = OrderedDict()
                local_token_id = 0
                for begin, end in be_indices:
                    assert begin != end
                    text_to_tokenize = line[begin:end]
                    assert text_to_tokenize != ''
                    tokens = tokenizer.tokenize(text_to_tokenize)
                    token_ids = []  # global
                    for i in range(len(tokens)):
                        global_token_id = len(all_tokens) + i
                        token_ids.append(global_token_id)
                        global_to_local_ids_map[global_token_id] = local_token_id
                        local_token_id += 1
                    token_ids_this_line += token_ids
                    all_tokens += tokens
                    tokens_this_line_dict[begin] = {'tokens': tokens, 'token_ids': token_ids}
                    tokens_this_line += tokens
                assert len(tokens_this_line) == len(token_ids_this_line)

                for node in nodes_to_process:
                    col_index = pc_links[node]['col']
                    assert col_index in tokens_this_line_dict
                    strong_token_global_id = tokens_this_line_dict[col_index]['token_ids'][0]
                    if FLAGS.pc_links_aug is None or FLAGS.pc_links_aug == 's_grease':
                        pc_links[node]['token_ids_global'] = [strong_token_global_id]  # 0th token id is taken
                        pc_links[node]['tokens'] = [tokens_this_line_dict[col_index]['tokens'][0]]
                        if FLAGS.pc_links_aug == 's_grease':
                            pc_links[node]['strong_weak_types'] = ['s']
                    elif FLAGS.pc_links_aug == 'all_line_sw':
                        pc_links[node]['token_ids_global'] = token_ids_this_line
                        pc_links[node]['tokens'] = tokens_this_line
                        pc_links[node]['strong_weak_types'] = ['w'] * len(token_ids_this_line)  # default: weak
                        strong_token_local_id = global_to_local_ids_map[strong_token_global_id]
                        pc_links[node]['strong_weak_types'][strong_token_local_id] = 's'
                    else:
                        assert FLAGS.pc_links_aug in ['all_line_swp', 'all_line_swp_grease']
                        if 'is_pragma' in pc_links[node]['ndata']:
                            assert pc_links[node]['ndata']['is_pragma']
                            pc_links[node]['token_ids_global'] = token_ids_this_line
                            pc_links[node]['tokens'] = tokens_this_line
                            pc_links[node]['strong_weak_types'] = ['p'] * len(token_ids_this_line)  # type: p
                        else:
                            pc_links[node]['token_ids_global'] = token_ids_this_line
                            pc_links[node]['tokens'] = tokens_this_line
                            pc_links[node]['strong_weak_types'] = ['w'] * len(token_ids_this_line)  # default: weak
                            strong_token_local_id = global_to_local_ids_map[strong_token_global_id]
                            pc_links[node]['strong_weak_types'][strong_token_local_id] = 's'
            elif FLAGS.pc_links_aug == 'grease':
                pass
            else:
                raise NotImplementedError()
        else:
            tokens_this_line = tokenizer.tokenize(line)
            all_tokens += tokens_this_line
        assert '\n' not in line  # need to have this additional \n tokenized as well
        all_tokens += tokenizer.tokenize('\n')  # TODO: debug at this point to see return type carefully
    return all_tokens, pc_links


def fit_preprocessors_src_code(preprocessors):
    if FLAGS.data_repr != 'ast':  # pure sequence
        return

    preprocessors['encoders']['enc_ntype'].fit(preprocessors['X_all']['X_ntype_all'])

    if len(preprocessors['X_all']['edge_ftype_all']) != 0:
        preprocessors['encoders']['enc_ftype_edge'].fit(preprocessors['X_all']['edge_ftype_all'])
    if len(preprocessors['X_all']['edge_ptype_all']) != 0:
        preprocessors['encoders']['enc_ptype_edge'].fit(preprocessors['X_all']['edge_ptype_all'])


''


def encode_X_torch_src_code(d_node, d_edge, preprocessors, gname, vname, return_Data=True):
    enc_ntype = preprocessors['encoders']['enc_ntype']
    enc_ftype_edge = preprocessors['encoders']['enc_ftype_edge']
    enc_ptype_edge = preprocessors['encoders']['enc_ptype_edge']
    g_new = d_node['g_new']

    # embeddings = []
    tokens_ids_list = []
    attention_masks = []
    for node, data in sorted(g_new.nodes(data=True)):
        tokens = data['tokens']
        if len(tokens) > MAX_LEN:
            raise ValueError(
                f'len(tokens)={len(tokens)} > MAX_LEN={MAX_LEN}; tokens={tokens}; Consider a larger max length')
        assert tokens is not None
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        # if FLAGS.token_att_masking: # should always generate attention_mask -- up to the model.py to decode to use it or not
        attention_mask = torch.tensor([1] * len(tokens_ids) + [0] * (MAX_LEN - len(tokens_ids))).view(1, -1)
        attention_masks.append(attention_mask)
        if len(tokens_ids) < MAX_LEN:
            tokens_ids += [tokenizer.pad_token_id] * (MAX_LEN - len(tokens_ids))
        for token in tokens_ids:
            assert token is not None
        tokens_ids = torch.tensor(tokens_ids).view(1, len(tokens_ids))
        tokens_ids_list.append(tokens_ids)
    X_node = torch.cat(tokens_ids_list, dim=0)

    X_ast_node_labels = torch.tensor([])
    X_ast_edge_labels = torch.tensor([])
    if FLAGS.data_repr == 'ast':
        # Encode additional node types.
        """
        x_dict is the returned dict by _encode_X_dict()
        """
        X_ntype = enc_ntype.transform(d_node['X_ntype'])
        X_ast_node_labels = torch.FloatTensor(X_ntype.toarray())

        X_edge_ftype = enc_ftype_edge.transform(d_edge['X_ftype'])
        if FLAGS.ptrans:
            X_edge_ptype = enc_ptype_edge.transform(d_edge['X_ptype'])
            from scipy.sparse import hstack

            X_ast_edge_labels = hstack((X_edge_ftype, X_edge_ptype))
            X_ast_edge_labels = coo_to_sparse(X_ast_edge_labels).to_dense()
        else:
            X_ast_edge_labels = torch.FloatTensor(X_edge_ftype.toarray())

    # if FLAGS.token_att_masking:
    attention_mask = torch.cat(attention_masks, dim=0)

    edge_index = create_edge_index(g_new)

    if return_Data:

        d_node.pop('X_pragma_nodes', None)  # somehow affects data loading with a key error, so remove it

        if FLAGS.task == 'regression':
            return Data(
                gname=gname,
                key=vname,
                x=X_node,
                edge_index=edge_index,
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
                xy_dict_programl=d_node,
                X_ast_node_labels=X_ast_node_labels,
                X_ast_edge_labels=X_ast_edge_labels,
                attention_mask=attention_mask,
                **d_node['X_pragma_trans']
            )
        elif FLAGS.task == 'class':
            return Data(
                gname=gname,
                key=vname,
                x=X_node,
                edge_index=edge_index,
                perf=d_node['perf'],
                xy_dict_programl=d_node,
                X_ast_node_labels=X_ast_node_labels,
                X_ast_edge_labels=X_ast_edge_labels,
                attention_mask=attention_mask,
                **d_node['X_pragma_trans']
            )
        else:
            raise NotImplementedError()

    else:
        return {'X_node': X_node, 'edge_index': edge_index, 'X_ast_node_labels': X_ast_node_labels,
                'X_ast_edge_labels': X_ast_edge_labels, 'attention_mask': attention_mask, 'd_node': d_node}

    # return X, [], edge_index, [], X_ast_node_labels, X_ast_edge_labels, attention_mask
