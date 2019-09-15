"""
functions related to kernels of persistence diagram
"""

import sys
import numpy as np
import time
import sklearn_tda as tda
from joblib import Parallel, delayed

from Esme.dgms.test import generate_swdgm
from Esme.helper.format import precision_format
from Esme.helper.time import timefunction
from Esme.helper.others import assert_names

def sw(dgms1, dgms2, kernel_type='sw', parallel_flag=False, **featkwargs):
    """
    :param dgms1: numpy array
    :param dgms2: numpy array
    :param parallel_flag:
    :param kernel_type:
    :param featkwargs: when kernel_type is sw, kwargs has n_d, bw
                    when kernel_type is pss, kwargs has bw
                     when kernel_type is wg, kwargs has bw, K,p
    :return: a computed kernel
    """
    def arctan(C, p):
        return lambda x: C * np.arctan(np.power(x[1], p))
    import signal

    def signal_handler(signum, frame):
        raise Exception("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    time_limit = max(int(len(dgms1)*len(dgms2)* 0.2),100)
    signal.alarm(time_limit)

    if parallel_flag==False:
        if kernel_type=='sw':
            assert_names(['n_directions', 'bw'], featkwargs)
            tda_kernel = tda.SlicedWassersteinKernel(num_directions=featkwargs['n_directions'], bandwidth=featkwargs['bw'])
        elif kernel_type=='pss':
            assert_names(['bw'], featkwargs)
            tda_kernel = tda.PersistenceScaleSpaceKernel(bandwidth=featkwargs['bw'])
        elif kernel_type == 'wg':
            assert_names(['bw', 'K', 'p'], featkwargs)
            tda_kernel = tda.PersistenceWeightedGaussianKernel(bandwidth=featkwargs['bw'], weight=arctan(featkwargs['K'], featkwargs['p']))
        elif kernel_type ==  'pf':
            assert_names(['bw', 'bwf'], featkwargs)
            # try:
            tda_kernel = tda.PersistenceFisherKernel(bandwidth_fisher=featkwargs['bwf'], bandwidth=featkwargs['bw'])
            # except RuntimeWarning:
                # tda_kernel = 0
                # print('RuntimeWarning catched')
                # print(f'dgm1', dgms1)
                # print(f'dgm2', dgms2)
        else:
            print ('Unknown kernel for function sw')

        diags1 = dgms1; diags2 = dgms2
        X = tda_kernel.fit(diags1)
        Y = tda_kernel.transform(diags2)
        Y = np.nan_to_num(Y) # todo wait for mathieu's reply
        return Y


@timefunction
def sw_parallel(dgms1, dgms2, kernel_type='sw', parallel_flag=True, granularity=25, **featkwargs):
    """
    build on top of function sw

    :param dgms1: a list of array.
    :param dgms2:
    :param kernel_type: sw, pss, wg
    :param parallel_flag: Ture if want to compute in parallel
    :param granularity: parameter for parallel computing.
    :param featkwargs: kwargs for sw/pss/wg
    :return:
    """

    t1 = time.time()
    assert_sw_dgm(dgms1)
    assert_sw_dgm(dgms2)
    n1 = len(dgms1); n2 = len(dgms2)
    kernel = np.zeros((n2, n1))

    if parallel_flag:
        # parallel version
        kernel = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(sw)(dgms1, dgms2[i:min(i+granularity, n2)], kernel_type=kernel_type,
                                                 **featkwargs) for i in range(0, n2, granularity))
        kernel = (np.vstack(kernel))
    else: # used as verification
        for i in range(n2):
            kernel[i] = sw(dgms1, [dgms2[i]], kernel_type=kernel_type, **featkwargs)

    t = precision_format(time.time()-t1, 1)
    print('Finish computing %s kernel of shape %s. Takes %s'%(kernel_type, kernel.shape, t))
    return (kernel/float(np.max(kernel)), t)

def sw_parallel_test():
    dgms1 = generate_swdgm(1000)
    dummy_kwargs = {'K':1, 'p':1}
    serial_kernel = sw_parallel(dgms1, dgms1, bw=1, parallel_flag=False, **dummy_kwargs)[0]
    parallel_kernel = sw_parallel(dgms1, dgms1, bw=1, parallel_flag=True, **dummy_kwargs)[0]
    diff = serial_kernel - parallel_kernel
    assert np.max(diff) < 1e-5

def assert_sw_dgm(dgms):
    # check sw_dgm is a list array
    # assert_sw_dgm(generate_swdgm(10))
    assert type(dgms)==list
    for dgm in dgms:
        try:
            if len(dgm) > 0:
                assert np.shape(dgm)[1]==2
            else:
                print('There exist empty dgm in dgms')
        except AssertionError:
            print('Not the right format for sw. Should be a list of array')

def random_dgms(n=10):
    dgm_list = []
    for i in range(n):
        dgm = np.random.random((10,2))
        dgm_list.append(dgm)
    return dgm_list


if __name__ == '__main__':
    dgms1 = generate_swdgm(100)
    # dummy_kwargs = {'n_directions':10, 'bw':1}
    dummy_kwargs = {'bwf':1, 'bw':1}
    print('serial kernel\n' + '-'*50)
    serial_kernel = sw_parallel(dgms1, dgms1, kernel_type='pf', parallel_flag=False, **dummy_kwargs)[0]
    print('parallel kernel\n' + '-'*50)
    parallel_kernel = sw_parallel(dgms1, dgms1, kernel_type='pf', parallel_flag=True, **dummy_kwargs)[0]
    print(serial_kernel, parallel_kernel)
    diff = serial_kernel - parallel_kernel
    print(f'max diff is {np.max(diff)}' )
    assert np.max(diff) < 1e-5
    sys.exit()

    dgms1 = random_dgms(300)
    kwargs = {'n_directions':10, 'bandwidth':1.0, 'K':1, 'p':1}
    print(sw_parallel(dgms1, dgms1, parallel_flag=False, kernel_type='sw', **kwargs))
