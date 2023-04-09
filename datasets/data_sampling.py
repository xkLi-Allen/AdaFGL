import torch
import random
import numpy as np
import scipy.sparse as sp
from random import choice
from torch import Tensor
from scipy.sparse import csr_matrix
from datasets.utils import idx_to_mask
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from datasets.add_edge import Ratio_Homo, Ratio_Hete
from datasets.utils import remove_duplicate_two_dimension_list_element


def data_partitioning(G, sampling, num_clients,
    ratio_train, ratio_val, ratio_test, 
    ratio_iso, ratio_homo, ratio_hete,
    structure_info_injection=True):
    num_nodes = G.num_nodes
    graph_nx = to_networkx(G, to_undirected=True)
    if sampling == 'Metis':
        import metispy as metis
        print("Conducting metis graph partition...")
        node_dict = {}
        n_cuts, membership = metis.part_graph(graph_nx, num_clients)
        for client_id in range(num_clients):
            client_indices = np.where(np.array(membership) == client_id)[0]
            client_indices = list(client_indices)
            node_dict[client_id] = client_indices
    elif sampling == 'Louvain':
        from datasets.structure_iid import structure_iid_louvain
        graph_nx_louvain = to_networkx(G, to_undirected=True)
        node_dict = structure_iid_louvain(graph=graph_nx_louvain, num_clients=num_clients)
        structure_info_injection = False
    subgraph_list = construct_subgraph_dict_from_node_dict(
        G=G,
        num_clients=num_clients,
        node_dict=node_dict,  
        graph_nx=graph_nx, 
        ratio_train=ratio_train,
        ratio_val=ratio_val,
        ratio_test=ratio_test,
        ratio_iso=ratio_iso,
        ratio_homo=ratio_homo,
        ratio_hete=ratio_hete,
        structure_info_injection=structure_info_injection
    )
    return subgraph_list


def construct_subgraph_dict_from_node_dict(num_clients, node_dict, G, graph_nx, 
    ratio_train, ratio_val, ratio_test, 
    ratio_iso, ratio_homo, ratio_hete,
    structure_info_injection = True):
    subgraph_list = []
    for client_id in range(num_clients):
        num_local_nodes = len(node_dict[client_id])
        local_node_idx = [idx for idx in range(num_local_nodes)]
        random.shuffle(local_node_idx)
        train_size = int(num_local_nodes * ratio_train)
        val_size = int(num_local_nodes * ratio_val)
        test_size = int(num_local_nodes * ratio_test)
        train_idx = local_node_idx[: train_size]
        val_idx = local_node_idx[train_size: train_size + val_size]
        test_idx = local_node_idx[train_size + val_size:]
        local_train_idx = idx_to_mask(train_idx, size=num_local_nodes)
        local_val_idx = idx_to_mask(val_idx, size=num_local_nodes)
        local_test_idx = idx_to_mask(test_idx, size=num_local_nodes)
        map_train_idx = []
        map_val_idx = []
        map_test_idx = []
        map_train_idx += [node_dict[client_id][idx] for idx in train_idx]
        map_val_idx   += [node_dict[client_id][idx] for idx in val_idx  ]
        map_test_idx  += [node_dict[client_id][idx] for idx in test_idx ]
        global_train_idx = idx_to_mask(map_train_idx, size=G.y.size(0))
        global_val_idx = idx_to_mask(map_val_idx, size=G.y.size(0))
        global_test_idx = idx_to_mask(map_test_idx, size=G.y.size(0))
        node_idx_map = {}
        edge_idx = []
        for idx in range(num_local_nodes):
            node_idx_map[node_dict[client_id][idx]] = idx
        edge_idx += [(node_idx_map[x[0]], node_idx_map[x[1]]) for x in graph_nx.subgraph(node_dict[client_id]).edges]
        edge_idx += [(node_idx_map[x[1]], node_idx_map[x[0]]) for x in graph_nx.subgraph(node_dict[client_id]).edges]
        if structure_info_injection:
            structure_info_inject = ["homo", "hete"]
            inject_way = choice(structure_info_inject)
            if inject_way == "homo":
                ratio, edge_idx = Ratio_Homo(edge_idx, G.y[node_dict[client_id]], ratio_homo, ratio_iso)
            elif inject_way == "hete":
                ratio, edge_idx = Ratio_Hete(edge_idx, G.y[node_dict[client_id]], ratio_hete, ratio_iso)
            edge_idx = remove_duplicate_two_dimension_list_element(edge_idx)
        edge_idx_tensor = torch.tensor(edge_idx, dtype=torch.long).T
        subgraph = Data(x=G.x[node_dict[client_id]],
                        y=G.y[node_dict[client_id]],
                        edge_index=edge_idx_tensor)
        subgraph.adj = sp.coo_matrix((torch.ones([len(edge_idx_tensor[0])]), (edge_idx_tensor[0], edge_idx_tensor[1])), shape=(num_local_nodes, num_local_nodes))
        subgraph.row, subgraph.col, subgraph.edge_weight = subgraph.adj.row, subgraph.adj.col, subgraph.adj.data
        if isinstance(subgraph.adj.row, Tensor) or isinstance(subgraph.adj.col, Tensor):
            subgraph.adj = csr_matrix((subgraph.edge_weight.numpy(), (subgraph.row.numpy(), subgraph.col.numpy())),
                                            shape=(subgraph.num_nodes, subgraph.num_nodes))
        else:
            subgraph.adj = csr_matrix((subgraph.edge_weight, (subgraph.row, subgraph.col)), shape=(subgraph.num_nodes, subgraph.num_nodes))
        subgraph.train_idx = local_train_idx
        subgraph.val_idx = local_val_idx
        subgraph.test_idx = local_test_idx
        subgraph.global_train_idx = global_train_idx
        subgraph.global_val_idx = global_val_idx
        subgraph.global_test_idx = global_test_idx
        subgraph_list.append(subgraph)
    return subgraph_list
