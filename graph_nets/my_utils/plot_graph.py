import networkx as nx
import matplotlib.pyplot as plt
from graph_nets.utils_np import graphs_tuple_to_networkxs

def plot_graph_structure(graphs_tuple):
    """ Plot the graph structure by converting the input graph into networkx graph """
    networkx_graphs = graphs_tuple_to_networkxs(graphs_tuple)
    num_graphs = len(networkx_graphs)
    _, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 5))
    if num_graphs == 1:
        axes = axes,
    for graph, ax in zip(networkx_graphs, axes):
        plot_graph_networkx(graph, ax)
    plt.show()

def plot_graph_networkx(graph, ax, pos=None):
    """ Given a graph(networkx-ed) and matplotlib axis components, this formats the resulting plot """
    node_labels = {node: "{:.3g}".format(data["features"][0])
                   for node, data in graph.nodes(data=True)
                   if data["features"] is not None}
    edge_labels = {(sender, receiver): "{:.3g}".format(data["features"][0])
                   for sender, receiver, data in graph.edges(data=True)
                   if data["features"] is not None}
    global_label = ("{:.3g}".format(graph.graph["features"][0])
                    if graph.graph["features"] is not None else None)

    if pos is None:
        pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, ax=ax, labels=node_labels)

    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax)

    if global_label:
        plt.text(0.05, 0.95, global_label, transform=ax.transAxes)

    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    return pos