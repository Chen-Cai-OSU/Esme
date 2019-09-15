import os
import sys
python="time /home/cai.507/anaconda3/bin/python"
file="/home/cai.507/Documents/DeepLearning/Esmé/Esme/dgms/fil.py"
chuck=200


def gen_args(idx, chuck, labels, parallel_flag = False):
    args = '--a ' +  str(idx*chuck) + ' --b ' + str(min((idx+1) *chuck, len(labels)))
    if parallel_flag: args += ' --parallel  --n_jobs=-1 '
    return args

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--parallel", action='store_true', help='use parallel or not')
parser.add_argument("--start", default=0, type=int, help='start chunk')
parser.add_argument("--end", default=-1, type=int, help='end chunk')

if __name__ == '__main__':
    from Esme.graph.dataset.modelnet import modelnet2graphs
    args = parser.parse_args()
    args.parallel = True
    _, labels = modelnet2graphs(version='40', print_flag=True, labels_only=True)
    n = len(labels)
    n_chucks = n // chuck
    print(n)
    end_chuck = n_chucks+1 if args.end == -1 else args.end

    for idx in range(end_chuck, args.start-1, -1):
        command = ' '.join([python, file, gen_args(idx, chuck, labels, parallel_flag = args.parallel), '--all'])
        print(command)
        os.system(command)