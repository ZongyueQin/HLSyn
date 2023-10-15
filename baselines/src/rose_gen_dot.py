from config import FLAGS
from saver import saver

from utils import print_stats, print_g
from subprocess import check_output, STDOUT, CalledProcessError

from tqdm import tqdm
from os.path import dirname, join, isfile, basename, splitext
from os import remove
import networkx as nx


def get_ast_graph(code_text, source_C_file, gname):
    dot_file_path = _get_dot_file(code_text, source_C_file, gname)

    if dot_file_path is None:
        return None

    return read_dot_file_as_graph(code_text, dot_file_path, gname)


def read_dot_file_as_graph(code_text, dot_file_path, gname):
    assert dot_file_path is not None
    g = _read_preproc_ast_graph(dot_file_path, code_text)
    g = nx.convert_node_labels_to_integers(g,
                                           ordering='sorted')  # super tricky! be careful; otherwise egde_index will be messed up
    # g.gname = gname
    p = f'{saver.get_obj_dir()}/{gname}_ast.gexf'
    nx.write_gexf(g, p)
    saver.log_info(f'Saved AST gexf to {p}')

    assert type(g) is nx.MultiDiGraph
    g = nx.DiGraph(
        g)  # critical: no need to use MultiDiGraph; DiGraph works better with later edge index generation code

    if FLAGS.bi_directional_AST:
        # critical: turns into an undirected graph with edge label
        g_new = g.copy()
        for nid1, nid2, edata in g.edges(data=True):
            g_new.add_edge(nid2, nid1, **edata)
            nx.set_edge_attributes(g_new, {(nid1, nid2): {"direction": 0}})
            nx.set_edge_attributes(g_new, {(nid2, nid1): {"direction": 1}})
        assert g_new.number_of_edges() == 2 * g.number_of_edges() # to debug, list(g_new.edges(data=True))
        assert len(g) == len(g_new)
        g = g_new

    g.gname = gname
    return g


def _read_preproc_ast_graph(dot_file, code_text):
    g = nx.drawing.nx_pydot.read_dot(dot_file)
    print_g(f'ast loaded from {dot_file}', g, print_func=saver.log_info)

    g_new = nx.MultiDiGraph()
    lines = code_text.split('\n')
    None_nodes = set()
    # print(g.number_of_nodes())
    for node, ndata in g.nodes(data=True):
        # print(ndata)
        g_new.add_node(node)
        assert 'label' in ndata, f'label not in ndata for {node}: {ndata}'
        nds = ndata['label'].split('\\n')
        attrs = {'shape': ndata.get('shape', '')}
        # try:
        attrs['type'] = nds[1] if len(nds) >= 2 else ''
        # except:
        #     print()

        # if 'Pragma' in ndata['label']:
        #     print()

        content = _find_content(nds, lines)
        if content == 'None':
            None_nodes.add(node)
        attrs['content'] = content
        # print(content)
        nx.set_node_attributes(g_new, {node: attrs})
    for nid1, nid2, edata in g.edges(data=True):
        elabel = edata['label']
        g_new.add_edge(nid1, nid2, elabel=elabel)

    while True:
        remove_nodes = set()
        for node in g_new.nodes():
            if len(list(g_new.neighbors(node))) == 0 and node in None_nodes:
                remove_nodes.add(node)
        for node in remove_nodes:
            g_new.remove_node(node)
        if not remove_nodes:
            break

    print_g(f'ast g_new after loading code content etc.', g_new, print_func=saver.log_info)
    return g_new


def _find_content(nds, lines):
    # f = strip_(f)
    for s_id, s in enumerate(nds):
        if 'compilerGenerated' in s or 'NULL_FILE' in s or 'SgSourceFile' in s:
            return 'None'
        if 'physical line=' in s:
            if '.cpp' in s or '.c' in s:
                line_num = int(s.split('physical line=')[1].split(')')[0])
                begin = int(s.split('raw line:col=')[1].split(')')[0].split(':')[1])
                # try:
                end = int(nds[s_id + 1].split('raw line:col=')[1].split(')')[0].split(':')[1])
                # except:
                #     print(nds)
                #     exit(-1)
                line = lines[line_num - 1]
                content = line[begin - 1:end]
                # print()
                return content
    return 'None'


# def strip_(s):
#     return s.replace('_', '')


def _get_dot_file(code_text, source_C_file, gname):
    ext = splitext(source_C_file)[1]
    dot_file_path = join(dirname(source_C_file), f'{gname}_rose{ext}.dot')

    # if gname == 'aes':
    #     return None # TODO: fix aes

    if isfile(dot_file_path):
        saver.log_info(f'dot_file_path found; load')
    else:
        saver.log_info(f'dot_file_path not found; generate using rose (docker -- ensure docker is started)')
        # Step 1: Save the temporary c file.
        code_text_processed = code_text.replace('#include "merlin_type_define.h"', '//#include "merlin_type_define.h"')
        temp_code_file_path = join(dirname(source_C_file), f'{gname}_rose{ext}')
        with open(temp_code_file_path, 'w') as f:
            f.write(code_text_processed)
        # Step 2: Run docker rose.
        assert isfile(temp_code_file_path)
        cmd = f'docker run --rm -v {dirname(source_C_file)}:/root gleisonsdm/rose dotGenerator -c {basename(temp_code_file_path)}'
        saver.log_info(cmd)
        output = None
        try:
            output = check_output(cmd, shell=True, stderr=STDOUT).decode('utf-8')
        except CalledProcessError as e:
            saver.log_info(f'Error of rose docker: {e.output}')
            raise e # just continue
        saver.log_info(output)
        try:
            remove(temp_code_file_path)
        except OSError:
            pass
        # exit()

    return dot_file_path


def main():
    cmd = 'docker run --rm -it -v $(pwd):/root gleisonsdm/rose dotGenerator -c main.cpp'

    print(cmd)
    output = check_output(cmd, shell=True).decode('utf-8')
    print(output)


if __name__ == '__main__':
    main()
