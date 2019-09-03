
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import numpy as np

def plot3dpts(x):
    """ viz a numpy array of shape (n, 3)"""

    assert x.shape[1] == 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], x[:,2])
    plt.show()

def plot_example(face, pos):
    """
    :param face: array of shape (3, n_face) (face index)
    :param pos: array coordinates of shape (n_pts, 3)
    :return:
    """

    assert face.shape[0] == pos.shape[1] == 3
    import plotly.graph_objects as go
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
            name='y',
            showscale=True
        )
    ])
    fig.show()

def plot_off(file='1'):
    from Esme.shape.util import off_face, off_pos
    face = off_face(file)
    pos = off_pos(file)
    print(face.shape, pos.shape)
    plot_example(face.T, pos)

if __name__ == '__main__':
    plot_off(file='1')
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
