def viz_tree(graph, name = 'wikipedia', show_flag=False):
    try:
        import pygraphviz
        from networkx.drawing.nx_agraph import graphviz_layout
    except ImportError:
        try:
            import pydot
            from networkx.drawing.nx_pydot import graphviz_layout
        except ImportError:
            raise ImportError("This example needs Graphviz and either PyGraphviz or pydot")

    for prog in ['sfdp', 'dot']:
        pos = graphviz_layout(graph, prog=prog, args='')
        nx.draw_networkx(graph, pos=pos, node_size=0.5, with_labels=False)
        import matplotlib.pyplot as plt
        plt.savefig('/home/cai.507/Documents/DeepLearning/gae/viz/' + str(name) + '_' + prog + '.png')
        # if show_flag:plt.show()
        plt.close()

dataset = 'texas'
# dataset = 'karate'
adj, features = load_data_(dataset)
graph = nx.from_scipy_sparse_matrix(adj)
print(nx.info(graph))
viz_tree(graph, dataset)
