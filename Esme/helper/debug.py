import numpy as np

def debug(x, name='x'):
    print('typf of %s is %s'%(name, type(x)))

    if type(x)==list:
        print('length of list %s is %s'%(name, str(len(x))))
        print('A single item in the list is ', x[0])
    if str(type(x)) ==  "<class 'numpy.ndarray'>":
        print('Shape of np.array is', x.shape)


if __name__=='__main__':
    gs = np.random.random((3,3))
    debug(gs, name='gs')