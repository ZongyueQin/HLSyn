#!/opt/conda/bin/python
from config import FLAGS
from data import get_data_list, MyOwnDataset
from data import update_save_dir
import data
from data_src_code import update_tokenizer
from train import train_main
from test import inference
from adapt import adapt_main
from saver import saver
from utils import OurTimer, load_replace_flags, load
from os.path import join, basename

import torch, numpy as np, random
import traceback

if FLAGS.fix_randomness:
    saver.log_info('Critical! Fix random seed for torch and numpy')
    torch.manual_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed(FLAGS.random_seed)

from torch import softmax

import config

TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL


class HandleNodeAttention(object):
    def __call__(self, data):
        data.attn = softmax(data.x[:, 0], dim=0)
        data.x = data.x[:, 1:]
        return data


timer = OurTimer()


def main():

    if FLAGS.load_model != 'None':
        saver.log_info(f'Loading model\'s config/FLAGS: {FLAGS.load_model}')
        load_replace_flags(FLAGS)
        saver.log_new_FLAGS_to_model_info()
        update_save_dir(FLAGS)
        update_tokenizer()
        print(basename(saver.logdir))

    if not FLAGS.force_regen:
        dataset = MyOwnDataset()
        pragma_dim = load(join(data.SAVE_DIR, 'pragma_dim'))
        print('read dataset')
    else:
        dataset, pragma_dim = get_data_list()
    saver.log_info(f'pragma_dim: {pragma_dim}')

    if len(dataset) == 0:
        raise ValueError('Empty dataset! Check config.py; Maybe use force_regen')
    saver.log_info(f'Dataset contains {len(dataset)} designs ')

    saver.log_info(f'dataset[0].num_features={dataset[0].num_features}')


    # if FLAGS.task == 'regression':
    if FLAGS.subtask == 'inference':
        if FLAGS.adaptation_needed:
            adapt_main(dataset, pragma_dim)
        inference(dataset, pragma_dim)
    else:
        train_main(dataset, pragma_dim)


if __name__ == '__main__':

    timer = OurTimer()

    try:
        main()
        status = 'Complete'
    except:
        traceback.print_exc()
        s = '\n'.join(traceback.format_exc(limit=-1).split('\n')[1:])
        saver.log_info(traceback.format_exc(), silent=True)
        saver.save_exception_msg(traceback.format_exc())
        status = 'Error'

    tot_time = timer.time_and_clear()
    saver.log_info(f'Total time: {tot_time}')
    saver.close()
