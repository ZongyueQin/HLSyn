from model import Model
from config import FLAGS
import torch

from collections import OrderedDict


class Code2VecNet(Model):
    def __init__(self, init_pragma_dict=None, dataset=None):
        super(Code2VecNet, self).__init__()
        self.task = FLAGS.task

        self.out_dim, self.loss_function = self._create_loss()

        self.target_list = self._get_target_list()

        self.decoder, _ = self._create_decoder_MLPs(384, 384, self.target_list,
                                                    self.out_dim, hidden_channels=None)

    def forward(self, data, forward_pairwise, tvt=None, epoch=None, iter=None, test_name=None):
        total_loss = torch.tensor(0.0, device=FLAGS.device)
        out_dict = OrderedDict()
        loss_dict = OrderedDict()

        self._apply_target_MLPs_with_loss(self.decoder, data.x.float(), data, total_loss, out_dict,
                                          loss_dict,
                                          'normal')

        return out_dict, total_loss, loss_dict, torch.tensor(0.0)
