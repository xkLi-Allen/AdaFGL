import torch
from config import args
from datasets.louvain.community import community_louvain
from sklearn.cluster import spectral_clustering


def structure_iid_louvain(graph,num_clients):
    num_nodes = graph.number_of_nodes() 
    partition = community_louvain.best_partition(graph)
    groups = []
    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    partition_groups = {group_i: [] for group_i in groups}
    for key in partition.keys():
        partition_groups[partition[key]].append(key)
    group_len_max = num_nodes // num_clients
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))
    len_dict = {}
    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)}
    owner_node_ids = {owner_id: [] for owner_id in range(num_clients)}
    owner_nodes_len = num_nodes // num_clients
    owner_list = [i for i in range(num_clients)]
    owner_ind = 0
    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) > owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        k = 0
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) > owner_nodes_len + 1:
            k += 1
            owner_ind = (owner_ind + 1) % len(owner_list)
            if k == len(owner_list):
                owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
                break
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
    node_dict = owner_node_ids
    return node_dict

def structure_iid_sc(num_nodes,F,num_clients):
    S = torch.sigmoid(torch.mm(F, F.T))
    S = S.cpu().detach().numpy()
    clustering_lbls = spectral_clustering(affinity=S, n_clusters=num_clients)
    clustering_lbls = clustering_lbls.tolist()
    node_dict = {client_id: [] for client_id in range(num_clients)}
    for node_idx in range(num_nodes):
        node_dict[clustering_lbls[node_idx]].append(node_idx)
    return node_dict