
# Imports
import networkx as nx


def construct_complete_graph(N):
    '''

    :param N: number of nodes in graph
    :return: G - the construct graph
             s - source node (0)
             t - sink node (N-1)
    '''

    # Create complete graph
    G = nx.complete_graph(N)
    G = nx.DiGraph(G)  # make it bidirectional

    # Set source and sink nodes
    s = 0
    t = N-1

    # Set edge capacities of graph to 0.5
    for e in G.edges(data=True):
        e[2]['capacity'] = 0.5

    return G, s, t

