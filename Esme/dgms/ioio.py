"""
functions related to input and output
"""

import os
import os.path as osp
from Esme.helper.io_related import make_dir
from Esme.dgms.format import dgms2diags, export_dgm, load_dgm, print_dgm
from Esme.helper.others import assert_names

def dgms_dir_test(**kwargs):
    """ return True if dgms have already been computed """
    kwargs.get('ntda', None) in [True, False, None]
    assert_names(['fil', 'graph', 'norm'], kwargs)
    dir = osp.dirname(osp.realpath(__file__))
    dir_ = osp.join(dir, '..', '..', 'save_dgms')

    try:
        dir = osp.join(dir_, kwargs['graph'], kwargs['fil'], kwargs['fil_d'],   'norm_' + str(kwargs['norm']), '')
    except:
        dir = osp.join(dir_, kwargs['graph'], kwargs['fil'], 'epd',             'norm_' + str(kwargs['norm']), '')

    # ugly hack. TODO: refactor
    if kwargs.get('ntda', None) == True:
        try:
            dir = osp.join(dir_, kwargs['graph'], kwargs['fil'], kwargs['fil_d'], 'ntda_True', 'norm_' + str(kwargs['norm']), '')
        except:
            dir = osp.join(dir_, kwargs['graph'], kwargs['fil'], 'epd',          'ntda_True',  'norm_' + str(kwargs['norm']), '')

    try:
        # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
        n =  len([name for name in os.listdir(dir) if name[-4:]=='.csv'])
    except:
        print('No existing csv files at %s'%dir)
        n = 0

    if n == 0:
        return dir, False
    else:
        return dir, True # there exists dgms

def save_dgms(dgms, **kwargs):
    # save dgms at a directory which is parameterized by kwargs
    assert kwargs.get('ntda', None) in [True, False, None]
    if kwargs.get('ntda', None) == True:
        pass
        # print('ntda = True so do not save dgms')

    dir, _ = dgms_dir_test(**kwargs)
    make_dir(dir)
    for i in range(len(dgms)):
        export_dgm(dgms[i], dir=dir, filename=str(i) + '.csv')
    print('Saving %s dgms at %s'%(len(dgms), dir))

def load_dgms( **kwargs):
    dir, _ = dgms_dir_test(**kwargs)
    dgms = []
    n =  len([name for name in os.listdir(dir) if name[-4:]=='.csv'])
    for i in range(n):
        dgm = load_dgm(dir=dir, filename=str(i)+'.csv')
        dgms.append(dgm)
    print('Loading existing %s dgms at %s'%(len(dgms), dir))
    return dgms

if __name__ == '__main__':
    from Esme.dgms.test import randomdgms
    dgms = randomdgms(10)
    kwargs = {'fil':'deg', 'fil_d':'sub', 'norm': False}
    save_dgms(dgms=dgms, **kwargs)
    dgms = load_dgms(**kwargs)
    print(dgms)
    print_dgm(dgms[7])

