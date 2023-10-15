from nn import MyGlobalAttention
from config import FLAGS
import torch
import torch.nn as nn
from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU
from utils import MLP

def create_graph_att_module(D, return_gate_nn=False):
    def _node_att_gate_nn(D):
        if FLAGS.node_attention_MLP:
            return MLP(D, 1,
                       activation_type=FLAGS.activation_type,
                       hidden_channels=[D // 2, D // 4, D // 8],
                       num_hidden_lyr=3)
        else:
            return Sequential(Linear(D, D), ReLU(), Linear(D, 1))

    gate_nn = _node_att_gate_nn(D)
    glob = MyGlobalAttention(gate_nn, None)
    if return_gate_nn:
        return gate_nn, glob
    else:
        return glob


class LpModule(torch.nn.Module):
    def __init__(self, p):
        super(LpModule, self).__init__()
        self.p = p

    def forward(self, x, **kwargs):
        x = x / (torch.norm(x, p=self.p, dim=-1).unsqueeze(-1) + 1e-12)
        return x
