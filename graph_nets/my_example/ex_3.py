from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

# @title #### (Define functions for generating and plotting graphs)

GLOBAL_SIZE = 4
NODE_SIZE = 5
EDGE_SIZE = 6


def get_graph_data_dict(num_nodes, num_edges):
    return {
        "globals": np.random.rand(GLOBAL_SIZE).astype(np.float32),
        "nodes": np.random.rand(num_nodes, NODE_SIZE).astype(np.float32),
        "edges": np.random.rand(num_edges, EDGE_SIZE).astype(np.float32),
        "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
        "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
    }


graph_3_nodes_4_edges = get_graph_data_dict(num_nodes=3, num_edges=4)
graph_5_nodes_8_edges = get_graph_data_dict(num_nodes=5, num_edges=8)
graph_7_nodes_13_edges = get_graph_data_dict(num_nodes=7, num_edges=13)
graph_9_nodes_25_edges = get_graph_data_dict(num_nodes=9, num_edges=25)

graph_dicts = [graph_3_nodes_4_edges, graph_5_nodes_8_edges,
               graph_7_nodes_13_edges, graph_9_nodes_25_edges]


def plot_graphs_tuple_np(graphs_tuple):
    networkx_graphs = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
    num_graphs = len(networkx_graphs)
    _, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 5))
    if num_graphs == 1:
        axes = axes,
    for graph, ax in zip(networkx_graphs, axes):
        plot_graph_networkx(graph, ax)


def plot_graph_networkx(graph, ax, pos=None):
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


def plot_compare_graphs(graphs_tuples, labels):
    pos = None
    num_graphs = len(graphs_tuples)
    _, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 5))
    if num_graphs == 1:
        axes = axes,
    pos = None
    for name, graphs_tuple, ax in zip(labels, graphs_tuples, axes):
        graph = utils_np.graphs_tuple_to_networkxs(graphs_tuple)[0]
        pos = plot_graph_networkx(graph, ax, pos=pos)
        ax.set_title(name)

graphs_tuple_tf = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)

print(graphs_tuple_tf)

OUTPUT_EDGE_SIZE = 10
OUTPUT_NODE_SIZE = 11
OUTPUT_GLOBAL_SIZE = 12
input_graphs = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)

graph_network = modules.GraphNetwork(
    edge_model_fn=lambda: snt.Linear(output_size=EDGE_SIZE),
    node_model_fn=lambda: snt.Linear(output_size=NODE_SIZE),
    global_model_fn=lambda: snt.Linear(output_size=GLOBAL_SIZE)
)

num_recurrent_passes = 3
previous_graphs = input_graphs

for _ in range(num_recurrent_passes):
    previous_graphs = graph_network(previous_graphs)
output_graphs = previous_graphs
print(output_graphs)

# plot_graphs_tuple_np(output_graphs)
# plt.show()