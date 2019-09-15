from torch_geometric.transforms import FaceToEdge
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import FaceToEdge


def load_modelnet(version='10', point_flag = False):
    assert version in ['10', '40']
    if point_flag:
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    else:
        pre_transform, transform =None, None

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet' + version)
    path = '/home/cai.507/Documents/DeepLearning/ModelNet' + version

    train_dataset = ModelNet(path, version, True, transform=transform, pre_transform=pre_transform)
    test_dataset = ModelNet(path, version, False, transform=transform, pre_transform=pre_transform)
    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = load_modelnet('10', point_flag=False)
    print(test_dataset[1])
    train_dataset, test_dataset = load_modelnet('40', point_flag=False)
    print(test_dataset[1])