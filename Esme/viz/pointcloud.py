import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import numpy as np
from Esme.shape.util import off_face, off_pos, off_face_color, off_face_color2


def plot3dpts(x):
    """ viz a numpy array of shape (n, 3)"""

    assert x.shape[1] == 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], x[:,2])
    plt.show()


def plot_example(face, pos, color=None, export_flag = False, fig_name = None):
    """
    :param face: array of shape (3, n_face) (face index)
    :param pos: array coordinates of shape (n_pts, 3)
    :return:
    """

    assert face.shape[0] == pos.shape[1] == 3
    import plotly.graph_objects as go
    n_face = face.shape[1]
    fig = go.Figure(data=[
        go.Mesh3d(
            x=pos[:,0].T,
            y=pos[:,1].T,
            z=pos[:,2].T,

            # i, j and k give the vertices of triangles
            # here we represent the 4 triangles of the tetrahedron surface
            i=face[0,:],
            j=face[1,:],
            k=face[2,:],
            colorbar_title='z',
            name='y',
            colorscale='algae',
            # facecolor = [(252.0, 141.0, 89.0)] * 3000 + [(255.0, 255.0, 191.0)]*3000 + [(145.0, 191.0, 219.0)]*(n_face - 6000), # np.random.randint(1, 3, size=int(face.shape[1])),# ['b'] * 3000 + ['r']*3000 + ['y']*(n_face - 6000), # np.random.randint(1, 256, size=int(face.shape[1])),
            facecolor = color, #['blue'] * 3000 + ['green'] * 3000 + ['red']*(n_face-6000) if
            showscale=True
        )
    ])
    fig.show()
    if export_flag:
        dir = os.path.join('/home/cai.507/Documents/DeepLearning/local-persistence-with-UF/images_2/')
        fig.write_image(dir + fig_name )


def plot_off(file='1', seg = '0'):
    face = off_face(file)
    pos = off_pos(file)
    # face_color = off_face_color(file=file, seg=seg, c_flag=True)
    face_color = off_face_color2(file=file, seg=seg, c_flag=True)
    print(face.shape, pos.shape)
    plot_example(face.T, pos, color=face_color, export_flag=False, fig_name=file + '_' + seg + '.png')


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--file", default=1, type=int, help='model index. Exclude models from 260 to 280')  # (1515, 500) (3026,)

if __name__ == '__main__':
    args = parser.parse_args()
    plot_off(file=str(args.file), seg = '0')
    sys.exit()


    pos = np.array([[0.1, 1, 2, 0], [0, 0, 1, 2], [0, 2, 0, 1]]).T
    face = np.array([[0,0,0,1], [1,2,3,2], [2,3,1,3]])
    print(pos.shape, face.shape)
    # print(face)
    # print(face[0,:])
    plot_example(pos = pos, face = face)
    sys.exit()

    # Download data set from plotly repo
    pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
    # x, y, z = pts.T
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
    z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
    fig.show()

    sys.exit()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
    z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    sys.exit()

    import numpy as np
    x = np.random.randint(low=1, high=100,size=(100,3))
    plot3dpts(x)

    sys.exit()
