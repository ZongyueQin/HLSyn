
from config import FLAGS
from saver import saver
from utils import get_root_path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, TransformerConv, GCNConv, GatedGraphConv
import torch.nn as nn

from collections import OrderedDict


class PretrainedGNNEncoder(nn.Module):
    def __init__(self, in_channels, edge_dim=0, init_pragma_dict=None, task=FLAGS.task, num_layers=FLAGS.num_layers,
                 D=64, target=FLAGS.target): # TODO: debug and check if edge dim/attr is used properly -- yes due to pretrained_model.load_state_dict loading the correct lin_edge
        super(PretrainedGNNEncoder, self).__init__()

        conv_class = TransformerConv # default choice so far (05/18/2023)

        self.conv_first = conv_class(in_channels, D, edge_dim=edge_dim)

        self.conv_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            conv = conv_class(D, D, edge_dim=edge_dim)
            self.conv_layers.append(conv)

        self.jkn = JumpingKnowledge(FLAGS.jkn_mode, channels=D, num_layers=2)

    def get_node_emb(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch

        if x is None:
            x, edge_index, edge_attr, batch = \
                data.x_programl, data.edge_index_programl, getattr(data, 'edge_attr_programl'), data.x_programl_batch

        # if hasattr(data, 'kernel'):
        #     gname = data.kernel[0]
        # print(gname)
        # print(edge_attr.shape)
        outs = []
        out_dict = OrderedDict()
        if FLAGS.activation == 'relu':
            activation = F.relu
        elif FLAGS.activation == 'elu':
            activation = F.elu
        else:
            raise NotImplementedError()

        out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))

        outs.append(out)

        for i, conv in enumerate(self.conv_layers):
            out = conv(out, edge_index, edge_attr=edge_attr)
            if i != len(self.conv_layers) - 1:
                out = activation(out)

            outs.append(out)

        if FLAGS.jkn_enable:
            out = self.jkn(outs)
        out = torch.tanh(out)
        node_emb = out

        return node_emb, outs


def create_and_load_pretrained_GNN():
    if not hasattr(FLAGS, 'pretrained_GNN_name') or FLAGS.pretrained_GNN_name == 'our':
        load_model = f'{get_root_path()}/file/xxx_pretrain/test_model_state_dict.pth'
    elif FLAGS.pretrained_GNN_name == 'GraphMAE':
        load_model = f'{get_root_path()}/file/xxx_pretrain/GraphMAE/test_model_state_dict.pth'
    else:
        raise NotImplementedError()

    pretrained_model = PretrainedGNNEncoder(in_channels=158, num_layers=6)
    ld = torch.load(load_model, map_location=torch.device('cpu'))
    # weights_cur = dict(pretrained_model.state_dict())
    pretrained_model.load_state_dict(ld, strict=False)
    saver.log_info(f'Loaded GNN encoder from {load_model}')
    return pretrained_model


if __name__ == '__main__':
    create_and_load_pretrained_GNN()
