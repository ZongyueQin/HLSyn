from config import FLAGS
from model import Net
from model_multi_modality import MultiModalityNet
from model_code2vec import Code2VecNet
from pretrained_GNN_handler import create_and_load_pretrained_GNN


def create_model(data_loader, pragma_dim):
    # saver.log_info(f'edge_dim={edge_dim}')
    # if FLAGS.dataset == 'simple-programl' or FLAGS.target_kernel is not None:
    #     init_pragma_dict = {'all': [1, 21]}
    if FLAGS.model == 'code2vec':
        c = Code2VecNet
    else:
        assert FLAGS.model == 'our'
        if FLAGS.multi_modality:
            c = MultiModalityNet
        else:
            c = Net
    model = c(init_pragma_dict=pragma_dim, dataset=data_loader.dataset).to(
        FLAGS.device)
    if FLAGS.load_pretrained_GNN:
        model.pretrained_GNN_encoder = create_and_load_pretrained_GNN().to(FLAGS.device)
    return model
