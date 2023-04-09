import numpy as np


def label_dirichlet_distribution(lbls_dict, num_clients, alpha=0.5):
    num_classes = len(lbls_dict)
    node_dict = {client_id:[] for client_id in range(num_clients)}
    for class_id in range(num_classes):
        num_nodes = len(lbls_dict[class_id])
        node_idx = lbls_dict[class_id]
        idx_per_client = [[]] * num_clients
        np.random.shuffle(node_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array(
            [p * (len(idx_j) < num_nodes / num_clients) for p, idx_j in zip(proportions, idx_per_client)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(node_idx)).astype(int)[:-1]
        idx_per_client = [idx_j + idx.tolist() for idx_j, idx in zip(idx_per_client, np.split(node_idx, proportions))]

        lbls_node_dict = {i: idx_per_client[i] for i in range(num_clients)}

        for client_id in range(num_clients):
            node_dict[client_id] += lbls_node_dict[client_id]
    return node_dict