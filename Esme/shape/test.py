import numpy as np

def choose(X, Y, method='node_label'):
    """

    :param X: a list of feats
    :param Y: a lisf of labels
    :param method:
    :return:
    """
    if method == 'node_label':  # mathieu's way
        face_label_dict = dict(zip(face_indices_tuple, y.tolist()))

        pt_labels = []
        for i in range(n_node):
            # filter a dict if i is in the keys
            face_label_dict_ = {k: v for k, v in face_label_dict.items() if i in k}

            # get corresponding labels
            labels = [face_label_dict[k] for k in face_label_dict_]
            lab = most_frequent(labels)
            pt_labels.append(lab)

        X.append(dgm_vector)
        Y.append(np.array(pt_labels))

    elif method == 'face_label':
        face_x = np.zeros((n_face, dgm_vector.shape[1]))
        for i in range(n_face):
            idx1, idx2, idx3 = face_indices[i]
            idx1, idx2, idx3 = int(idx1), int(idx2), int(idx3)
            face_x[i, :] = dgm_vector[idx1][:] + dgm_vector[idx2, :] + dgm_vector[idx3, :]
        print(face_x.shape, y.shape)
        X.append(face_x)
        Y.append(y)

    else:
        raise Exception(f'No such method {method}')

    return X, Y
