import networkx as nx
from networkx.linalg.algebraicconnectivity import fiedler_vector
import numpy.random as random
import sys
from Esme.graph.OllivierRicci import ricciCurvature

if __name__ == '__main__':

    # a bad example for fiedler
    lines = ["0 1 {'dist': 1.3296398}", "0 2 {'dist': 0.9401972}", "0 3 {'dist': 0.94019735}","1 2 {'dist': 0.94019735}","1 3 {'dist': 0.9401972}","1 4 {'dist': 0.9402065}","3 4 {'dist': 1.3296462}","3 5 {'dist': 0.9402065}","4 5 {'dist': 0.9401972}"]
    g = nx.parse_edgelist(lines, nodetype=int)
    print(g.edges(data=True))
    # v_weight = fiedler_vector(g, normalized=False, weight='dist')  # np.ndarray
    # v_weight = list(nx.closeness_centrality(g, distance='dist').values())

    g = ricciCurvature(g, alpha=0.5, weight='dist')
    ricci_dict = nx.get_node_attributes(g, 'ricciCurvature')
    v_weight= [ricci_dict[i] for i in range(len(g))]

    print(v_weight)
    sys.exit()


    # g = nx.circulant_graph(10, offsets=[1]*10)
    w_name = 'weightd'
    random.seed(43)
    g = nx.random_tree(20, seed=42)
    for u, v in g.edges():
        g[u][v][w_name] = random.random()
        # print(g[u][v])
    print(g.edges)

    v_noweight = fiedler_vector(g, normalized=False)  # np.ndarray
    v_weight = fiedler_vector(g, normalized=False, weight=w_name)  # np.ndarray
    v_fake_weight = fiedler_vector(g, normalized=False, weight='abcdefg')  # np.ndarray
    print('no weight', v_noweight)
    print('weight' ,v_weight)
    print('fake weight', v_fake_weight)
