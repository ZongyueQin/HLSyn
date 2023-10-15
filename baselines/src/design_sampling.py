from model import get_y_with_target
from config import FLAGS
from saver import saver
from torch_geometric.data import DataLoader
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch_scatter import scatter_add
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

if FLAGS.task == 'regression':
    TARGETS = ['perf', 'actual_perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
elif FLAGS.task == 'class':
    TARGETS = ['perf']
else:
    raise NotImplementedError()

class DesignSampler(object):
    def __init__(self, data_list, sample_algo, K, model, embs=None):

        if FLAGS.load_db != 'None':
            assert len(data_list) > 0
            self.data_list = data_list
            return

        self.data_list = data_list
        self.data_list_indices = list(range(len(data_list)))
        self.sample_algo = sample_algo
        assert K <= len(self.data_list), \
            f'Not enough designs in data list {len(self.data_list)} -- try decreasing K {K}'
        self.K = K

        if 'special' in sample_algo:
            ss = sample_algo.split('special_')[1].split('_')
            self.special_tag, self.special_ratio = ss[0], float(ss[1])
            saver.log_info(f'Special sampling looking for design keys containing '
                           f'{self.special_tag} with ratio {self.special_ratio}')
            return

        if 'load' in sample_algo:
            load_path = sample_algo.split('load_')[1]
            lines = []
            with open(load_path, 'r') as f:
                for l in f:
                    l = l.rstrip()
                    if l:
                        lines.append(l)
            self.load_keys = set(lines)
            saver.log_info(f'Design sampler: Read {len(lines)} lines '
                           f'({len(self.load_keys)} '
                           f'keys) from {load_path}')
            return
        if sample_algo == 'random':
            key = 'gemb'
            n_neighbors = len(self.data_list)
        else:
            ss = sample_algo.split('_')
            self.s_algo = ss[0]
            assert self.s_algo in ['KNN', 'spread', 'spreaddot', 'spreadcosine',
                                   'greedy', 'greedydot', 'greedycosine', 'greedycosinesmallest', 'volume']
            key = ss[1]
            assert key in ['X', 'gemb'], 'X -- init features; gemb -- learned'
            assert model is not None
            n_neighbors = None
            if self.s_algo == 'KNN':
                n_neighbors = K
            elif self.s_algo == 'spread':
                n_neighbors = len(self.data_list)

        self.embs = self._get_or_encode_embs(data_list, key, model, embs)

        if self.embs is not None:
            self.tot_distances = euclidean_distances(self.embs)
            self.tot_cosine = cosine_similarity(self.embs)
            self.tot_dot = np.tensordot(self.embs, self.embs, axes=(1, 1))
            if sample_algo == 'random' or self.s_algo in ['KNN', 'spread', 'greedy']:
                if n_neighbors is not None:
                    self.nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.embs)
                    self.distances, self.indices = self.nbrs.kneighbors(self.embs)
                # self.tot_distances = euclidean_distances(self.embs)
            elif self.s_algo in ['greedydot', 'spreaddot', 'greedycosine', 'greedycosinesmallest', 'spreadcosine']:
                    # self.tot_dot = np.tensordot(self.embs, self.embs, axes=(1, 1))
                    if self.s_algo in ['greedycosine', 'greedycosinesmallest', 'spreadcosine']:
                        if self.s_algo == 'spreadcosine':
                            self.indices = np.argsort(self.tot_cosine, axis=1)
                    if self.s_algo == 'spreaddot':
                        self.indices = np.argsort(self.tot_dot, axis=1)
        self.model = model

        self.sample_time = 0
        # exit()

    def _get_or_encode_embs(self, data_list, key, model, embs):
        if embs is None:
            if model is not None:
                _, embs, _ = encode_data_list(
                    dataset=data_list, only_kernel='None', batch_size=128, model=model,
                    vis_what='tsne', vis_emb=key, vis_emb_P_or_T='T', vis_y='target')
        else:
            assert embs.shape[0] == len(data_list)
        return embs

    def sample_K(self):
        if FLAGS.load_db != 'None':
            saver.log_info(f'FLAGS.load_db is not None -- design sampler simply returns'
                           f' all stuff ({len(self.data_list)}) in loaded db')
            return self.data_list, 0

        if self.K == -1:
            return self.data_list, None
        if 'load' in self.sample_algo or 'special' in self.sample_algo:
            rtn = []
            for data in self.data_list:
                if 'load' in self.sample_algo:
                    if data.key in self.load_keys:
                        rtn.append(data)
                else:
                    if self.special_tag in data.key.decode('utf-8'):
                        rtn.append(data)
            l = len(rtn)
            rtn = rtn[0:int(len(rtn) * self.special_ratio)]
            saver.log_info(f'Take {len(rtn)} ({self.special_ratio}) out of {l}')
            if 'load' in self.sample_algo:
                if len(rtn) > len(self.load_keys):
                    raise ValueError(f'Expect {len(self.load_keys)}'
                                     f' but {len(rtn)} data')
            else:
                if len(rtn) == 0:
                    saver.log_info(f'No design with tag {self.special_tag}!')
            return rtn, 0 # TODO: quality computation
        if self.sample_algo == 'random':
            random.Random(self.sample_time).shuffle(self.data_list_indices)
            indices = self.data_list_indices[0:self.K]
        else:
            ind = self.sample_time % len(self.data_list)
            if self.s_algo in ['KNN', 'spread', 'spreaddot', 'spreadcosine']:
                indices = self.indices[ind].tolist()
                if self.s_algo == 'KNN':
                    pass
                elif self.s_algo in ['spread', 'spreaddot', 'spreadcosine']:
                    indices = self._take_equal_space_elements(indices)
                else:
                    assert False
            elif self.s_algo == 'greedy':
                indices = self._greedy_select_K(self.tot_distances, ind, 'max')
            elif self.s_algo == 'greedydot':
                indices = self._greedy_select_K(self.tot_dot, ind, 'min')
            elif self.s_algo == 'greedycosine':
                indices = self._greedy_select_K(self.tot_cosine, ind, 'min')
            elif self.s_algo == 'greedycosinesmallest':
                li = []
                for ind in range(len(self.data_list)):
                    indices = self._greedy_select_K(self.tot_cosine, ind, 'min')
                    quality = self._get_avg_of_submatrix(self.tot_cosine, indices)
                    li.append((quality, indices))
                li.sort()
                assert len(li) == len(self.data_list)
                indices = li[0][1]
            else:
                assert self.s_algo == 'volume'
                # from maxvolpy.maxvol import rect_maxvol, py_rect_maxvol
                from my_maxvol import rect_maxvol, py_rect_maxvol

                # piv, C = py_rect_maxvol(self.embs, 1.0, minK=self.K, maxK=self.K)
                # indices = piv[0:self.K].tolist()
                # exit()

                indices = self._greedy_select_K_volume(ind)
        take = [self.data_list[i] for i in indices]

        if self.model is not None:
            if FLAGS.adapt_designs_sample_algo_quality == 'avg_dist':
                quality = self._get_avg_of_submatrix(self.tot_distances, indices)
            elif FLAGS.adapt_designs_sample_algo_quality in ['avg_dot', 'avg_cosine']:
                to_get = 'tot_dot' \
                    if 'dot' in FLAGS.adapt_designs_sample_algo_quality else 'tot_cosine'
                if not hasattr(self, to_get):
                    raise ValueError(
                        f'Request {FLAGS.adapt_designs_sample_algo_quality} '
                        f'but not using it for sampling:'
                        f' need {to_get}')
                mat = getattr(self, to_get)
                quality = self._get_avg_of_submatrix(mat, indices)
            elif FLAGS.adapt_designs_sample_algo_quality == 'volume':
                A = self.embs[indices]
                assert A.shape == (self.K, self.embs.shape[1])
                quality = self._compute_volume(A)
            else:
                raise ValueError(f'Unrecognized {FLAGS.adapt_designs_sample_algo_quality}')
        else:
            quality = None

        self.sample_time += 1
        return take, quality

    def _get_avg_of_submatrix(self, mat, indices):
        submatrix = mat[np.ix_(indices, indices)]
        assert submatrix.shape == (self.K, self.K)
        return np.mean(submatrix)


    def _take_equal_space_elements(self, li):
        indices = np.linspace(0, len(li) - 1, self.K)
        indices = [int(x) for x in indices]
        take = [li[x] for x in indices]
        return take

    def _greedy_select_K(self, mat, ind, minmax):
        rtn = [ind]

        m, n = mat.shape
        assert m == n == self.embs.shape[0]
        while len(rtn) < self.K:
            put = []
            # Find the index with the smallest total dot products
            # or largest total distances w.r.t.
            # current indices (greedy).
            for j in range(n):
                temp = mat[rtn,j]
                assert temp.shape == (len(rtn),)
                tot_sum = np.sum(temp)
                put.append((tot_sum, j))
            if minmax == 'min':
                reverse = False
            elif minmax == 'max':
                reverse = True
            else:
                raise ValueError(f'{minmax}')
            put.sort(reverse=reverse)
            select = put[0]
            # if select[1] == ind:
            #     # raise ValueError(f'Trying to select {put[0]} w.r.t. {ind}')
            #     select = put[1]
            #     print('@@@', ind, put)
            #     exit()
            if select[1] in rtn: # tricky: could have found a design already in rtn
                put_id = 1
                while select[1] in rtn:
                    select = put[put_id]
                    put_id += 1
            assert select[1] != ind
            rtn.append(select[1])

        assert len(rtn) == self.K and len(set(rtn)) == self.K, f'rtn={rtn}; ' \
            f'set(rtn)={set(rtn)}({len(set(rtn))})'
        return rtn

    def _greedy_select_K_volume(self, ind):
        rtn = [ind]

        m, n = self.embs.shape
        while len(rtn) < self.K:
            put = []
            # Find the index with the largest total volume w.r.t.
            # current indices (greedy).
            for j in range(m):
                temp_li = rtn + [j]
                A = self.embs[temp_li]
                volume = self._compute_volume(A)
                put.append((volume, j))
            put.sort(reverse=True)
            rtn.append(put[0][1])

        assert len(rtn) == self.K
        return rtn

    def _compute_volume(self, A):
        # A_p = A.T.dot(A)
        # quality = np.linalg.det(A_p)
        m, n = A.shape
        assert m < n, f'usually take K=10 embeddings out of D={n} but now m={m}'
        A_p = A.dot(A.T)
        volume = np.linalg.det(A_p)
        return volume


def encode_data_list(dataset, only_kernel, batch_size, model,
                     vis_what, vis_emb, vis_emb_P_or_T, vis_y):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    embs = []
    ys_dict = defaultdict(list)

    design_names = []

    for i, data in enumerate(tqdm(data_loader)):
        assert len(data.gname) == len(data.key)
        for x, y, p in zip(data.gname, data.key, data.perf):
            if only_kernel == 'None' or only_kernel in x:
                design_names.append(f'{x}_{y}_perf={p:.4f}')
        # print()
        if vis_what == 'att':
            model(data.to(FLAGS.device), tvt='all', epoch=0, iter=i)
        elif vis_what == 'tsne':
            if vis_emb == 'X':
                size = int(data.batch.max().item() + 1)
                X = scatter_add(data.x, data.batch, dim=0, dim_size=size)
                embs.append(_filter(X.detach().cpu().numpy(), data, only_kernel))
            elif vis_emb == 'gemb':
                out_dict, *_ = model(data.to(FLAGS.device), tvt='all', epoch=0, iter=i)
                # print(data)
                # print(out_dict['emb'].shape)
                embs.append(
                    _filter(out_dict[f'emb_{vis_emb_P_or_T}'].detach().cpu().numpy(), data,
                            only_kernel))
            else:
                raise NotImplementedError()
            if vis_y == 'kernel_name':
                y = _filter([name.split('_')[0] for name in data.gname], data, only_kernel)
                ys_dict['Kernel'] += y
            elif vis_y == 'target':
                for target_name in TARGETS:
                    y = get_y_with_target(data, target_name).detach().cpu().numpy()
                    ys_dict[target_name].append(_filter(y.reshape(len(y), 1), data, only_kernel))
            else:
                raise NotImplementedError(f'Unrecognized {vis_y}')
        else:
            raise NotImplementedError()

    return design_names, np.vstack(embs), ys_dict


def _filter(arr, data, only_kernel):
    if only_kernel != 'None':
        rtn = []
        for i, gname in enumerate(data.gname):
            if only_kernel in gname:
                take = arr[i]
                take = take.reshape(1, len(take))
                # print(f'take {take.shape} {arr.shape}')
                rtn.append(take)
                # exit()
        if len(rtn) != 0:
            rtn = np.vstack(rtn)
        else:
            rtn = np.array([]).reshape(0, arr.shape[1])
        # print(rtn.shape)
        return rtn
    else:
        return arr
