# from importlib import reload  # Python 3.4+ only.
import matplotlib
matplotlib.use('tkagg')
import collections
import sys
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

def viz_graph(g, node_size = 5, edge_width = 1, node_color = 'b', color_bar = False, show = False):
    # g = nx.random_geometric_graph(100, 0.125)
    pos = nx.spring_layout(g, seed=1)
    nx.draw(g, pos, node_color=node_color, node_size=node_size, with_labels=False, width = edge_width)
    if color_bar:
        # https://stackoverflow.com/questions/26739248/how-to-add-a-simple-colorbar-to-a-network-graph-plot-in-python
        sm = plt.cm.ScalarMappable( norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
        sm._A = []
        plt.colorbar(sm)
    plt.title(f'graph of {len(g)}/{len(g.edges)}')
    if show: plt.show()

def test():
    G = nx.star_graph(20)
    pos = nx.spring_layout(G)
    colors = range(20)
    cmap = plt.cm.Blues
    vmin = min(colors)
    vmax = max(colors)
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors, width=4, edge_cmap=cmap,
            with_labels=False, vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.show()

def sample():
    G = nx.random_geometric_graph(200, 0.125)
    pos = nx.get_node_attributes(G, 'pos')

    # find node near center (0.5,0.5)
    dmin = 1
    ncenter = 0
    for n in pos:
        x, y = pos[n]
        d = (x - 0.5)**2 + (y - 0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d

    # color by path length from node near center
    p = dict(nx.single_source_shortest_path_length(G, ncenter))

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=list(p.keys()),
                           node_size=8,
                           node_color=list(p.values()),
                           cmap=plt.cm.Reds_r)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('off')
    plt.show()

def viz_deghis(G):
    # G = nx.gnp_random_graph(100, 0.02)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()

def viz_mesh_graph(edge_index, pos, viz_flag = True):
    """
    :param edge_index: np.array of shape (2, 2*num_edge)
    :param pos: np.array of shape (n_pts, 3)
    :return: viz mesh graph
    """
    n_node = pos.shape[0]
    n_edge = edge_index.shape[1] // 2

    edges = edge_index # np.array([[ 0,  0,  1,  1 ], [ 1,  9,  0,  2]])
    edges_lis = list(edges.T)
    edges_lis = [(edge[0], edge[1]) for edge in edges_lis]

    pos_dict = dict()
    for i in range(n_node):
        pos_dict[i] = tuple(pos[i,:])

    g = nx.from_edgelist(edges_lis)
    for node, value in pos_dict.items():
        g.node[node]['pos'] = value
    assert len(g) == n_node
    assert len(g.edges()) == n_edge

    if not viz_flag: return g

    nx.draw(g, pos_dict)
    plt.show()

def generate_random_3Dgraph(n_nodes, radius, seed=None):
    if seed is not None:
        random.seed(seed)

    # Generate a dict of positions
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}

    # Create random 3D network
    G = nx.random_geometric_graph(n_nodes, radius, pos=pos)

    return G

def network_plot_3D(G, angle, save=False, feat = None):
    # https://www.idtools.com.au/3d-network-graphs-python-mplot3d-toolkit/
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    if feat == None:
        edge_max = max([G.degree(i) for i in range(n)])
        colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
    else:
        assert len(feat) == n
        feat_max = max(abs(feat))
        colors = [plt.cm.plasma(feat[i] / feat_max) for i in range(n)]

    # 3D network plot
    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=1, edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.2)

    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    if save is not False:
        # plt.savefig("C:\scratch\\data\"+str(angle).zfill(3)+".png  ")
        plt.close('all')
    else:
        plt.show()
    return

if __name__=='__main__':
    import networkx as nx
    # g = nx.random_geometric_graph(1000, 0.1)
    g = generate_random_3Dgraph(n_nodes=100, radius=0.25, seed=1)
    network_plot_3D(g, 0, save=False)

    sys.exit()
    n = 100
    edge_index = np.array([[0, 0, 1, 1, 2, 2],
                           [1, 2, 0, 2, 0, 1]])
    pos = np.array([[1, 2, 3],
                    [2.1, 3, 5],
                    [1, 1, 2.2]])
    # pos = pos[:, 0:2]

    g= viz_mesh_graph(edge_index, pos, viz_flag=False) # generate_random_3Dgraph(n_nodes=n, radius=0.25, seed=1)

    sys.exit()
    viz_mesh_graph(edge_index, pos)
    sys.exit()
    test()
