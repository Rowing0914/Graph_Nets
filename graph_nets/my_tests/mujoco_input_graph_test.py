"""
test if we can construct the input graph from the obs of actual mujoco env using for loop.
"""
import gym
import tensorflow as tf
from graph_nets import modules
from graph_nets.my_utils.input_graph_generator import GraphOperator
from graph_nets.my_utils.mujoco_parser import parse_mujoco_graph

tf.compat.v1.enable_eager_execution()

env_name = "CentipedeThree-v1"
node_info = parse_mujoco_graph(task_name=env_name)  # get node_info
Graph_Operator = GraphOperator(input_dict=node_info["input_dict"],
                               output_list=node_info["output_list"],
                               obs_shape=node_info["debug_info"]["ob_size"])

env = gym.make(env_name)

# create model
graph_network = modules.GraphNetworkEager(internal_timestep=3,
                                          edge_layer_sizes=[16, 16, 1],
                                          node_layer_sizes=[16, 16, 1],
                                          global_layer_sizes=[16, 16, 1])
state = env.reset()
done = False
import time
for t in range(1000):
    # while not done:
    # env.render()
    begin = time.time()
    input_graph = Graph_Operator.obs_to_graph(state)
    now = time.time()
    print("obs to graph ", now - begin)
    begin = now
    output_graph = graph_network(input_graph)
    now = time.time()
    print("graph computation ", now - begin)
    begin = now
    action = Graph_Operator.readout_action(output_graph).numpy()
    now = time.time()
    print("readout ", now - begin)
    begin = now
    state, reward, done, info = env.step(action)
    now = time.time()
    print("env.step ", now - begin)
