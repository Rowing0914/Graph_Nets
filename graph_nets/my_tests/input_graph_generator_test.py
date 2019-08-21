"""
This tests the class of "InputGraphGenerator" which converts the obs from MuJoCo simulation into an input_graph
And to check the graph structure, we plot the input_graph by actually feeding the fake input!
"""

import numpy as np
from graph_nets.my_utils.input_graph_generator import GraphOperator
from graph_nets.my_utils.mujoco_parser import parse_mujoco_graph
from graph_nets.my_utils.plot_graph import plot_graph_structure

node_info = parse_mujoco_graph(task_name="WalkersHopperthree-v1") # get node_info
Graph_Operator = GraphOperator(input_dict=node_info["input_dict"],
                               output_list=node_info["output_list"],
                               obs_shape=node_info["debug_info"]["ob_size"])
obs = np.random.random(node_info["debug_info"]["ob_size"]).astype(np.float32) # create a fake input
input_graph = Graph_Operator.obs_to_graph(obs) # convert the obs into an input graph
# print(input_graph) # checker
print(input_graph.nodes)
print(input_graph.edges)
print(input_graph.receivers)
print(input_graph.senders)
# plot_graph_structure(input_graph) # plot the graph structure
