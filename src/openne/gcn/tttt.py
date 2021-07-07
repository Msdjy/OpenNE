import networkx as nx

def plot_graph(G):
    '''
    G: a networkx G
    '''
    # % matplotlib
    # notebook
    import matplotlib.pyplot as plt
    # plt.clf()
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()

    nodelist1 = []
    nodelist2 = []
    for i in range(34):
        if G.nodes[i]['club'] == 'Mr. Hi':
            nodelist1.append(i)
        else:
            nodelist2.append(i)
    nx.draw_networkx(G, pos, edges=edges);

    nx.draw_networkx_nodes(G, pos, nodelist=nodelist1, node_size=300, node_color='r', alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist2, node_size=300, node_color='b', alpha=0.8)
    # nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.4)
    plt.show()


G = nx.karate_club_graph()
plot_graph(G)
