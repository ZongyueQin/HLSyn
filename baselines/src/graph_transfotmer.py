from config import FLAGS
from torch_geometric.data import Batch
from torch_geometric.utils import degree, get_laplacian, to_scipy_sparse_matrix
import torch
from scipy.sparse.linalg import eigs, eigsh
import numpy as np


# CACHE = {}

def collate_batch_graph_transformer(data_li):
    X_list = []
    lengths = []
    assert type(data_li) is list and len(data_li) >= 1
    for data in data_li:
        x_list = [data.x]
        lengths.append(data.x.shape[0])

        for encoding in FLAGS.graph_transformer_option['graph_encoding']:
            if encoding == 'degree':
                x_list += _degree(data)
            elif encoding == 'Laplacian':
                x_list.append(_Laplacian_eigen(data))
            else:
                raise NotImplementedError()

        x = torch.cat(x_list, dim=1)
        data.x = x  # critical: this enables gps conv to leverage data.x in model's forward
        X_list.append(x)

    attention_map_aug = FLAGS.graph_transformer_option.get('attention_map_aug')
    att_mask_tensor = None
    if attention_map_aug is not None:
        att_mask_list = []
        for data in data_li:
            if attention_map_aug == 'proximity':
                att_mask_list.append(_proximity_att_mask(data))
            else:
                raise NotImplementedError()
        att_mask_tensor = _pad_list_of_2D_square_matrices(att_mask_list).to(FLAGS.device)

    X_padded = torch.nn.utils.rnn.pad_sequence(X_list, batch_first=True).to(
        FLAGS.device)  # trick: use torch's function to pad node input features

    batch = Batch.from_data_list(data_li).to(
        FLAGS.device)  # trick: use torch geometric's function to create a batch object
    batch.X_padded = X_padded
    batch.X_padded_lengths = lengths
    batch.att_mask_tensor = att_mask_tensor

    return batch


def _pad_list_of_2D_square_matrices(list_of_mat):
    lengths = []
    for mat in list_of_mat:
        shape_mat = mat.shape
        assert len(shape_mat) == 2 and shape_mat[0] == shape_mat[1]
        lengths.append(shape_mat[0])
    assert len(lengths) >= 1
    max_len = max(lengths)
    new_list_of_mat = []
    for mat in list_of_mat:
        new_mat = torch.nn.functional.pad(mat, (0, max_len - mat.shape[0], 0, max_len - mat.shape[0]))
        assert new_mat.shape == (max_len, max_len)
        new_list_of_mat.append(new_mat)
    return torch.stack(new_list_of_mat)


def get_num_features_graph_transformer(num_features):
    tot = num_features
    for encoding in FLAGS.graph_transformer_option['graph_encoding']:
        if encoding == 'degree':
            tot += 2
        elif encoding == 'Laplacian':
            tot += FLAGS.graph_transformer_option['Laplacian_K']
        else:
            raise NotImplementedError()
    return tot


def _degree(data):
    out_degrees = degree(data.edge_index[0])
    in_degrees = degree(data.edge_index[1])
    return [in_degrees.view(-1, 1), out_degrees.view(-1, 1)]


def _Laplacian_eigen(data):
    # Reference: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/add_positional_encoding.html#AddLaplacianEigenvectorPE
    is_undirected = FLAGS.graph_transformer_option['is_undirected']
    k = FLAGS.graph_transformer_option['Laplacian_K']
    eig_fn = eigs if not is_undirected else eigsh
    # The method eigsh does not check the input for being Hermitian;
    # # it just follows a process assuming it is; so the output is incorrect when the assumption fails.
    # eigs takes a lot longer to run than eigsh

    num_nodes = data.num_nodes
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization='sym',
        num_nodes=num_nodes,
    )

    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    eig_vals, eig_vecs = eig_fn(
        L,
        k=k + 1,
        which='SR' if not is_undirected else 'SA',
        return_eigenvectors=True,
    )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:k + 1])
    sign = -1 + 2 * torch.randint(0, 2, (k,))  # tricky: see https://arxiv.org/pdf/2003.00982.pdf
    pe *= sign
    return pe


def _proximity_att_mask(data):
    num_nodes = data.num_nodes
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization='sym',
        num_nodes=num_nodes,
    )

    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    return torch.tensor(L.todense())


'''
Below: Master branch of https://github.com/pyg-team/pytorch_geometric (as of 03/02/2023)
'''

import inspect
from typing import Any

import torch.nn.functional as F
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing

from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

from typing import Dict, List
from typing import Any, Optional, Union

import torch
from torch import Tensor


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(classes: List[Any], class_dict: Dict[str, Any],
             query: Union[Any, str], base_cls: Optional[Any],
             base_cls_repr: Optional[str], *args, **kwargs):
    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ''
    base_cls_repr = normalize_string(base_cls_repr)

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


# Activation Resolver #########################################################


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def activation_resolver(query: Union[Any, str] = 'relu', *args, **kwargs):
    import torch
    base_cls = torch.nn.Module
    base_cls_repr = 'Act'
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)


# Normalization Resolver ######################################################


def normalization_resolver(query: Union[Any, str], *args, **kwargs):
    import torch

    import torch_geometric.nn.norm as norm
    base_cls = torch.nn.Module
    base_cls_repr = 'Norm'
    norms = [
        norm for norm in vars(norm).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)


class GPSConv(torch.nn.Module):
    r"""The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper.

    The GPS layer is based on a 3-part recipe:

    1. Inclusion of positional (PE) and structural encodings (SE) to the input
       features (done in a pre-processing step via
       :class:`torch_geometric.transforms`).
    2. A local message passing layer (MPNN) that operates on the input graph.
    3. A global attention layer that operates on the entire graph.

    .. note::

        For an example of using :class:`GPSConv`, see
        `examples/graph_gps.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        graph_gps.py>`_.

    Args:
        channels (int): Size of each input sample.
        conv (MessagePassing, optional): The local message passing layer.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        dropout (float, optional): Dropout probability of intermediate
            embeddings. (default: :obj:`0.`)
        attn_dropout (float, optional): Dropout probability of the normalized
            attention coefficients. (default: :obj:`0`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
    """

    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            act: str = 'relu',
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout

        self.attn = torch.nn.MultiheadAttention(
            channels,
            heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        # print('x.shape', x.shape)
        # print('batch.shape', batch.shape)
        h, mask = to_dense_batch(x, batch)

        # print('mask.shape', mask.shape)
        # print('h.shape', h.shape)

        h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                h = self.norm3(h, batch=batch)
            else:
                h = self.norm3(h)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')
