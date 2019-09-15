import yaml
import os
import sys
from itertools import product
import argparse
from Esme.applications.motif.NRL.src.classification import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_experiment_config(config_path):
    with open(config_path, 'r') as conf:
        return yaml.load(conf, Loader=yaml.FullLoader)

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--feat", default='sw', type=str, help='sw/pss/wg')
parser.add_argument("--graph", default='mutag', type=str, help='mutag, imdb_binary...')
parser.add_argument("--fil", default='random', type=str, help='deg, ricci')
parser.add_argument("--flip", default='False', type=str, help='True, False')
parser.add_argument("--epd", default='False', type=str, help='True, False')
parser.add_argument("--permute", default='False', type=str, help='True, False')
parser.add_argument("--ntda", default='False', type=str, help='True, False')
parser.add_argument("--n_cv", default=1, type=int, help='number of cv')


if __name__=='__main__':
    # Handle the parameters when calling replicate.py
    args = parser.parse_args()
    feat=args.feat
    graph = args.graph
    fil = args.fil
    dir = os.path.dirname(__file__)
    print(dir)

    feat_dict = get_experiment_config(os.path.join(dir, feat + '.yaml')) # fead_dict = {'bw': {'format': 'values', 'values': [0.01, 0.1, 1], 'dtype': 'float'},...}

    dict_ = {}
    for key in feat_dict.keys():
        tmp = {key: feat_dict[key]['values']}
        dict_ = {**dict_, **tmp}

    python = "/home/cai.507/anaconda3/bin/python "
    dir = "/home/cai.507/Documents/DeepLearning/Esmé/Esme/permutation/replicate.py "
    command = python + dir + 'with graph=' + graph \
              + ' fil=' + fil + ' ' \
              + 'feat=' + feat + ' ' \
              + 'flip=' + args.flip + ' ' \
              + 'epd=' + args.epd + ' ' \
              + 'permute=' + args.permute + ' ' \
              + 'ntda='    + args.ntda + ' ' \
              + 'n_cv='    + str(args.n_cv) + ' '

    for d in product_dict(**dict_):
        suffix = ''
        for k in d.keys():
            suffix += 'feat_kwargs.' + str(k) + '=' + str(d[k]) + ' ' # append feat_kwargs.key=val
        print(command + suffix)
        os.system(command + suffix)