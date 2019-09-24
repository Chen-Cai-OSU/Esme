import os
import pickle
from persUF import geodesic_diagrams_on_shape
import sys
import numpy as np
from Esme.dgms.format import diag2dgm
from Esme.helper.io_related import make_dir
from Esme.helper.time import timefunction
from collections import Counter
DGM_DIRECT = '/home/cai.507/Documents/DeepLearning/local-persistence-with-UF/dgms/'
SEG_DIRECT = '/home/cai.507/Documents/DeepLearning/meshdata/MeshsegBenchmark-1.0/data/seg/Benchmark/'
LAB_DIRECT = '/home/cai.507/Documents/DeepLearning/local-persistence-with-UF/labeledDb/LabeledDB_new/'
# off file format:
# https://en.wikipedia.org/wiki/OFF_(file_format)

def loady(model=1, seg=0, counter= True):
    """
    load labels for face for segmentation task

    :param model: model idx
    :param seg: use 0 (human segmentation) as default.
    :return: a ndarray of shape (n_face, )
    """

    # read ground truth from benchmark
    direct = os.path.join(SEG_DIRECT, str(model), str(model) + '_' + str(seg) + '.seg')
    print(direct)
    with open(direct, 'rb') as f:
        y = f.readlines()
    y = [int(i) for i in y]
    assert len(y) == face_num(str(model))
    if counter: print(f'label counter {Counter(y)}')
    return np.array(y)

def load_labels(cat='Airplane', idx=61):
    direct = os.path.join(LAB_DIRECT, cat, str(idx) + '_labels.txt')
    with open(direct, 'rb') as f:
        y = f.readlines() # y is a list when odd index is category and even idx is faces that belong to that category

    n = len(y)
    assert n % 2 == 0
    cats, indices = [], []
    for i in range(n):
        if i % 2 == 0:
            cats.append(y[i][:-2])
        else:
            indices.append(y[i])

    max_faceidx, min_faceidx = 0, 10000
    for indice in indices:
        indice = str(indice).split(' ')[:-1]
        indice = [generalized_int(i) for i in indice]
        if max_faceidx < max(indice):
            max_faceidx = max(indice)
        if min_faceidx > min(indice):
            min_faceidx = min(indice)

    assert min_faceidx == 1
    res = np.zeros(max_faceidx) - 1
    labels = list(range(len(cats)))
    unsort_cats = cats.copy()
    cats.sort()

    try:
        cats2labels_dict = parts_dict()[cat]
    except:
        cats2labels_dict = dict(zip(cats, labels))

    print(cats2labels_dict)

    for i, indice in enumerate(indices):
        indice = str(indice).split(' ')[:-1]
        indice = [generalized_int(idx) for idx in indice]
        cat = unsort_cats[i]
        label = cats2labels_dict[cat]
        # print(idx, cat,label),
        for k in indice:
            res[k - 1] = label
    # print('-'*150)
    return res

def node_num(file='0', allflag=False, print_flag = False):
    if int(float(file)) in list(range(260, 281)): sys.exit()

    file = os.path.join(DGM_DIRECT, '..','tmp', '') + file
    with open(file + '.off') as f:
        first_line = f.readline()
        assert first_line == 'OFF\n'
        second_line = f.readline()

    if allflag: return second_line
    n =  int(second_line.split(' ')[0])
    if print_flag: print('file %s has %s nodes'%(file, n))
    return n

def face_num(file='0'):
    res = node_num(file, allflag=True)
    return int(res.split(' ')[1])

@timefunction
def face_idx(file = '1'):
    """ read face index of the off file for shap segmentation"""
    file_ = os.path.join(DGM_DIRECT, '..', 'tmp', '') + file
    with open(file_ + '.off') as f:
        res = f.readlines()

    n_node, n_face = node_num(file), face_num(file)
    assert len(res) == n_node + n_face + 2
    face_indices = res[n_node+2: ] # a list of face index ('3 1676 468 3987\n')
    assert len(face_indices) == n_face

    final_indices = []
    for indice in face_indices:
        tmp = indice.split('\n')[0].split(' ') # ['3', '1185', '1183', '1184']
        tmp = [int(i) for i in tmp]
        final_indices.append(tmp[1:])
    return final_indices

def off_pos(file='1'):
    # return the position of a off file
    file_ = os.path.join(DGM_DIRECT, '..', 'tmp', '') + file
    with open(file_ + '.off') as f:
        res = f.readlines()

    n_node, n_face = node_num(file), face_num(file)
    assert len(res) == n_node + n_face + 2
    pos = res[2: 2+n_node]
    pos_list = []
    for pt in pos:
        tmp = pt.split('\n')[0].split(' ')  # ['3', '1185', '1183', '1184']
        tmp = [float(i) for i in tmp]
        pos_list.append(tmp)
    res = np.array(pos_list)
    assert res.shape == (n_node, 3)
    return res

def off_face(file='1'):
    face_indices = face_idx(file)
    return np.array(face_indices)

def color_map():
    res = {0: 'blue',
           1: 'yellow',
           2: 'green',
           3: 'red',
           4: 'purple',
           5: 'black',
           6: 'orange',
           7: 'grey'
           }
    return res

def off_face_color(file='1', seg = '0', c_flag = False):
    file = os.path.join(SEG_DIRECT, file, file + '_' + seg + '.seg')

    with open(file) as f:
        face_color = f.readlines()

    face_color = [int(c.split('\n')[0]) for c in face_color]
    cmap = color_map()
    if c_flag:
        face_color = [cmap.get(c,'white') for c in face_color]
    return face_color

def off_face_color2(file='1', seg='0', c_flag=False):
    # file = '1'
    cat = get_cat(int(file))
    face_color = load_labels(cat=cat, idx=int(file)) # facecolor is a list of number so far
    cmap = color_map()
    if c_flag: face_color = [cmap.get(c, 'white') for c in face_color]
    return face_color

def vdgms2fdgms(dgms):
    pass

def savedgm(res, f):
    direct = DGM_DIRECT
    direct = os.path.join(direct,  f)
    make_dir(direct)
    with open(os.path.join(direct,'m'+f +'.pkl'), 'wb') as fp:
        pickle.dump(res, fp)

@timefunction
def loaddgm(f, form = 'mathieu', print_flag = True):
    """
    :param f:
    :param form: either mathieu's form or dionysus's form
    :return:
    """
    assert form in ['mathieu', 'dionysus']

    direct = DGM_DIRECT
    direct = os.path.join(direct,  f)
    direct = os.path.join(direct, 'm' + f + '.pkl')
    if print_flag: print(f'loading from {direct}')
    with open(direct, 'rb') as fp:
        dgms = pickle.load(fp)

    if form == 'mathieu':
        return dgms # # a typical dgm is x the following form: x = ([(1834, 1), (2434, 2416)], [(1.596, 0.0), (1.53, 0.493)])
    else:
        dio_dgms = []  # a list of pd (array of shape (n,2))
        for dgm in dgms:
            idx_list, pd_list = dgm[0], dgm[1]
            assert len(idx_list) == len(pd_list)
            pd = np.array(pd_list)
            assert pd.shape[1] == 2
            dio_dgms.append(pd)
        dgms = [diag2dgm(diag) for diag in dio_dgms]
        return dgms

def check(f):
    res_ = loaddgm(f)
    file =  f + '.off'
    n = node_num(file=f)
    res = geodesic_diagrams_on_shape(file, range(n))
    assert res == res_

def modeidx(low=1000, upper=10000):
    res = []
    for f in range(1, 400):
        n = node_num(file=str(f))
        if low < n and n < upper: res.append(f)
    print('There are %s modes whose node is from %s to %s'%(len(res), low, upper))
    for i in res:
        print(i),
    return res

def exist_file(f):
    try:
        res = loaddgm(f)
        n = node_num(file=f)
        if len(res)==n:
            return True
        else:
            return False
    except IOError:
        return False

def prince_cat():

    # categories of princeton shape benchmark
    cats = ['Human', 'Cup', 'Glasses', 'Airplane', 'Ant', 'Chair', 'Octopus', 'Table', 'Teddy', 'Hand', 'Plier', 'Fish', 'Bird', 'Spring',
            'Armadillo', 'Bust', 'Mech', 'Bearing', 'Vase', 'Fourleg']
    assert len(cats) == 20
    start_indices = list(range(1, 382, 20))
    end_indices = [i+20 for i in start_indices]
    keys = [ (start_indices[i], end_indices[i]) for i in range(20)]
    cat_dict = dict(zip(keys, cats))
    return cat_dict

def generalized_int(i):
    assert type(i) == str
    try:
        return int(i)
    except:
        return int(i.split("b'")[1])

def get_cat(idx):
    for k, v in prince_cat().items():
        if idx >= k[0] and idx < k[1]:
            print(f'idx {idx} is {v}')
            return v
    raise Exception(f'No cat for idx {idx} found')

def parts_dict():
    """ encode parts dict for different (not all) shapes """
    parts_dict = {}
    parts_dict['Fourleg'] = {b'earhorn': 0, b'head': 1, b'leg': 2, b'neck': 3, b'tail': 4, b'torso': 5}
    parts_dict['Vase'] = {b'body': 0, b'handle': 1, b'spout': 2, b'top': 3,  b'base':4}
    parts_dict['Bearing'] = {b'base1': 0, b'base2': 1, b'bigroller': 2, b'smallroller': 3, b'tinyroller': 4, b'mediumroller': 5}
    parts_dict['Mech']  = {b'bigbox': 0, b'smallcylinder': 1, b'mediumcylinder': 2, b'smallbox': 3,  b'bigcylinder':4}
    parts_dict['Bust'] = {b'background': 0, b'ear': 1, b'eye': 2, b'face': 3, b'hair': 4, b'mouth': 5, b'neck': 6, b'nose': 7}
    parts_dict['Armadillo'] = {b'back': 0, b'ear': 1, b'foot': 2, b'hand': 3, b'head': 4, b'lowerArm': 5, b'lowerLeg': 6, b'tail': 7, b'torso': 8, b'upperArm': 9, b'upperLeg': 10}
    parts_dict['Bird'] = {b'body': 0, b'head': 1, b'leg': 2, b'tail': 3, b'wing': 4}
    parts_dict['Fish'] = {b'body': 0, b'fin': 1, b'tail': 2}
    parts_dict['Plier'] = {b'center': 0, b'edge': 1, b'handle': 2}
    parts_dict['Hand'] = {b'finger1': 0, b'finger2': 1, b'finger3': 2, b'finger4': 3, b'hand': 4, b'thumb': 5}
    parts_dict['Teddy'] = {b'ear': 0, b'hand': 1, b'head': 2, b'leg': 3, b'torso': 4}
    parts_dict['Airplane'] = {b'body': 0, b'engine': 1, b'rudder': 2, b'stabilizer': 3, b'wing': 4}
    parts_dict['Glasses'] = {b'lens': 0, b'middle': 1, b'skeleton': 2}
    parts_dict['Cup'] = {b'body': 0, b'handle': 1}
    parts_dict['Human'] = {b'foot': 0, b'hand': 1, b'head': 2, b'lowerarm': 3, b'lowerleg': 4, b'torso': 5, b'upperarm': 6, b'upperleg': 7}
    return parts_dict



if __name__ == '__main__':
    from Esme.dgms.stats import stat, dgms_summary
    from itertools import chain
    import matplotlib.pyplot as plt

    num_pdpoints = []
    for i in range(1,5): #chain(range(1, 261), range(281,401)):
        dgms = loaddgm(str(i), form = 'dionysus', print_flag = True)
        print(get_cat(i))
        _, stat = dgms_summary(dgms) # stat is (mean, std, min, max)
        num_pdpoints.append(stat[0])
    plt.plot(stat)
    plt.show()



    sys.exit()
    res = face_idx(file='1')
    print(res)
    sys.exit()
    cat_dict = prince_cat()

    cat = 'Human'
    for idx in chain(range(1, 261), range(281,401)):
        for k, v in cat_dict.items():
            if k[0]<= idx and k[1]>idx:
                cat = v
                break
        print(cat)
        y = load_labels(cat=cat, idx=idx)
        if idx % 20==0: print('-'*50)

    sys.exit()

    print(off_face_color(c_flag=True))
    print(off_face_color(c_flag=False))

    for i in range(1, 2):
        file = str(i)
        contents = face_idx(file)
        print(contents[-10:])
        n_node, max_faceidx = node_num(file), face_num(file)
        assert len(contents) == n_node + max_faceidx + 2
    # print('num of nodes', node_num(file))
    # print('num of faces', face_num(file))

    # print(node_num('1'))