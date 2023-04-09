import numpy as np
import math


def uniform_distribution(num_clients, lbls_dict):
    node_dict = {client_id:[] for client_id in range(num_clients)}
    num_classes = len(lbls_dict)
    for class_id in range(num_classes):
        num_nodes = len(lbls_dict[class_id])
        proportions = np.random.uniform(low=0, high=num_clients, size=num_nodes)
        for node_idx in range(num_nodes):
            node_dict[math.floor(proportions[node_idx])].append(lbls_dict[class_id][node_idx])
    return node_dict