import operator
import tensorflow as tf
from functools import reduce
from graph_nets.utils_tf import data_dicts_to_graphs_tuple_eager


class GraphOperator:
    """
    this class deals with converts the obs into a graph by using the node info provided by "mujoco_parser.py"

    input_dict: {0: [0, 1, 5, 6, 7], 1: [2, 8], 2: [3, 9], 3: [4, 10]} <= sample
    """

    def __init__(self, input_dict, output_list, obs_shape):
        self.num_nodes = obs_shape
        self._create_nodes_indices(input_dict)
        self._create_action_indices(output_list)
        self._define_send_receive(input_dict)
        self._create_edges(obs_shape) # TODO: what values should be edges
        self._create_globals()

        self.readout_func = tf.keras.layers.Dense(self.num_action, activation="tanh")

    def _create_globals(self):
        self.globals = tf.random.normal(shape=(1,), dtype=tf.float32) # TODO: what value should be globals
        self.num_globals = len(self.globals)

    def _create_nodes_indices(self, input_dict):
        """ Given the input obs, we order it and treat as a feature/features of a node """
        temp_indices = [v for k, v in input_dict.items()]
        self._node_indices = reduce(operator.concat, temp_indices)
        self.num_nodes = len(self._node_indices)

    def _create_action_indices(self, output_list):
        """ Create the action indices to readout the action from processed nodes features """
        self._action_indices = [self._node_indices.index(v) for v in output_list]
        self.num_action = len(self._action_indices)

    def _create_edges(self, obs_shape):
        """ Define edges which express the relation of two nodes(sender and receiver) """
        self.edges = tf.random.uniform(minval=-0.3, maxval=0.3, shape=(obs_shape, 1), dtype=tf.float32)
        self.num_edges = obs_shape

    def _define_send_receive(self, input_dict):
        """ Defines the relationship between senders and receivers on edges """
        self.senders, self.receivers = list(), list()
        for key, value in input_dict.items():
            for v in value:
                # sen -> rec on one edge
                # self.senders.append(key)
                # self.receivers.append(v)
                self.senders.append(v)
                self.receivers.append(key)

    def obs_to_graph(self, obs):
        """Convert obs into a graph
        output: input_graph which contains the all relations among nodes and edges
        """
        nodes = tf.compat.v1.gather(tf.compat.v1.cast(obs, dtype=tf.float32), self._node_indices) # Ordering the obs
        nodes = tf.compat.v1.reshape(nodes, (self.num_nodes, 1))

        data_dict = {
            "globals": self.globals,
            "nodes": nodes,
            "edges": self.edges,
            "receivers": self.receivers,
            "senders": self.senders
        }
        return data_dicts_to_graphs_tuple_eager([data_dict])

    @tf.contrib.eager.defun(autograph=False)
    def readout_action(self, graph):
        """ Convert the resulting graph into an action """
        action_elems = tf.compat.v1.reshape(tf.compat.v1.gather(graph.nodes, self._action_indices), [-1])
        action = tf.compat.v1.reshape(self.readout_func(tf.compat.v1.expand_dims(action_elems, 0)), [-1])
        return action