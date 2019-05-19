import argparse
from Esme.dgms.arithmetic import add_dgms
from Esme.dgms.format import assert_dgms_above, assert_dgms_below, flip_dgms


def combine_dgms(subdgms = None, supdgms = None, epddgms = None, ss = False, epd = False, flip=False):
    """
    :param subdgms: a list of subdgms
    :param supdgms: a list of supdgms
    :param epddgms: a list of epddgms
    :param ss: If True, combine both sub and super level filtration
    :param epd: If True, also add extended persistence diagram
    :param flip: If True, flip superdgms above the diagonal
    :return: a list of dgms
    """

    assert len(subdgms) == len(supdgms) == len(epddgms)
    assert_dgms_above(subdgms)
    assert_dgms_below(supdgms)
    assert_dgms_above(epddgms)
    if flip: supdgms = flip_dgms(supdgms)

    if ss:
        dgms = add_dgms(subdgms, supdgms)
        if epd:
            dgms = add_dgms(dgms, epddgms)
    else:
        dgms = subdgms
    return dgms

if __name__ == '__main__':
    # argparse seems unnecessary after using Sacred

    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('group1')
    group1.add_argument('--graph', type=str, default='mutag', help="graph dataset",
                        choices=['mutag', 'ptc', 'nci1', 'reddit_binary'])
    lis1 = ['--graph', 'mutag']

    group2 = parser.add_argument_group('group2')
    group2.add_argument('--fil', type=str, default='ricci', help="Filtration function",
                        choices=['deg', 'ricci', 'random'])
    group2.add_argument('--norm', action='store_true', help="Whether to normalize filtration function")
    group2.add_argument('--epd', action='store_true', help="Use extended persistent homology")
    group2.add_argument('--ss', action='store_true',
                        help="Use both sublevel and superlevel filtration. Otherwise only sublevel")
    lis2 = ['--fil', 'ricci', '--norm']

    group3 = parser.add_argument_group('group3')
    group3.add_argument('--permute', action='store_true', help='Whether apply permutation test on computed dgm')
    group3.add_argument('--feat', type=str, default='sw', help="sw, pss, pi")
    lis3 = ['--feat', 'sw']

    group4 = parser.add_argument_group('group4')
    group4.add_argument('--classifier', type=str, default='svm', help="svm, rf, eigenpro",
                        choices=['svm', 'rf', 'eigenpro'])
    group4.add_argument('--n_cv', type=int, default='10', help="number of 10 cross validations")
    group4.add_argument('--bw', type=float, default=1, help="Bandwidth for sw/pss")
    lis4 = ['--n_cv', '10']
    args = parser.parse_args(lis1 + lis2 + lis3 + lis4)  # interactive test
    # args = parser.parse_args()
    arg_groups={}

    for group in parser._action_groups:
        group_dict= {a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)