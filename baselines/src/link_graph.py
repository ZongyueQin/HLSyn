from utils import get_root_path
from saver import saver

from os.path import basename
from pathlib import Path

def find_ast(programl_gexf_path):
    bn = basename(programl_gexf_path)
    source_path = Path(f'{get_root_path()}/dse_database/machsuite/dot-files/original-size')
    # source_path = Path(f'{get_root_path()}/dse_database/machsuite/sources/original-size')
    source_path_polly = Path(f'{get_root_path()}/dse_database/polly/dot-files')

    if 'gemm-blocked' in bn:
        return Path(f'{source_path}/gemm-blocked_kernel.c.gexf')
    elif 'gemm-ncubed' in bn:
        return Path(f'{source_path}/gemm-ncubed_kernel.c.gexf')
    elif 'stencil_stencil2d' in bn:
        return Path(f'{source_path}/stencil_kernel.c.gexf')
    elif 'aes' in bn:
        return Path(f'{source_path}/aes_kernel.c.gexf')
    elif 'nw' in bn:
        return Path(f'{source_path}/nw_kernel.c.gexf')
    elif 'spmv-crs' in bn:
        return Path(f'{source_path}/spmv-crs_kernel.c.gexf')
    elif 'spmv-ellpack' in bn:
        return Path(f'{source_path}/spmv-ellpack_kernel.c.gexf')
    elif 'atax' in bn:
        return Path(f'{source_path_polly}/atax_kernel.c.gexf')
    elif 'mvt' in bn:
        return Path(f'{source_path_polly}/mvt_kernel.c.gexf')
    else:
        saver.log_info(f'Cannot find ast gexf for {programl_gexf_path}')
        return None