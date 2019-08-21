"""
test the mujoco parser func which is supposed to extract the necessary info and output the dict
which contains the info below

dict(tree,
     relation_matrix
     node_type_dict,
     output_type_dict,
     input_dict,
     output_list,
     debug_info,
     node_parameters,
     para_size_dict,
     num_nodes)

"""

from graph_nets.my_utils.mujoco_parser import parse_mujoco_graph

res = parse_mujoco_graph(task_name="WalkersHopperthree-v1")
# print(res)

for key, item in res.items():
    print("========")
    print(key, item)
