
# Imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_sol(G, pos, P):
    '''

    :param G: graph object
    :param pos: layout of graph nodes
    :param P: solved picos problem
    :return:
    '''
    # Unpack variables from picos problem -P
    x = {}
    for e in G.edges():
        x[e] = P.variables['x[{0}]'.format(e)]

    # Determine which edges carry flow
    flow_edges = [e for e in G.edges() if x[e].value > 1e-4]

    # Draw the nodes and the edges that don't carry flow
    fig = plt.figure()
    nx.draw_networkx(G, pos, edge_color='lightgrey',
                     edgelist=[e for e in G.edges
                               if e not in flow_edges and (e[1], e[0]) not in flow_edges])

    # Draw the edges that carry flow
    nx.draw_networkx_edges(G, pos, edgelist=flow_edges)

    # Show flow values and capacities on these edges.
    labels = {e: '{0}'.format(np.round(x[e].value, 4)) for e in flow_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Show the maximum flow value.
    fig.suptitle("Minimum cost value: {}".format(P.obj_value()), fontsize=16, y=0.95)

