import networkx as nx
from datasets.dirichlet_distribution import dirichlet_distribution




def structure_non_iid_node(graph, num_clients):
    num_nodes = graph.number_of_nodes()
    node_dict = dirichlet_distribution(num_entity=num_nodes, num_assignments=num_clients)
    return node_dict
    
def structure_non_iid_edge(graph, num_clients):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edges = list(graph.edges)
    edge_dict = dirichlet_distribution(num_entity=num_edges, num_assignments=num_clients)
    node_dict = {client_id:set() for client_id in range(num_clients)}
    for client_id in range(num_clients):
        for edge_idx in edge_dict[client_id]:
            node_dict[client_id].add(edges[edge_idx][0])
            node_dict[client_id].add(edges[edge_idx][1])
    cnt = [0] * num_nodes
    remain_nodes = []
    for client_id in range(num_clients):
        for node_idx in node_dict[client_id]:
            cnt[node_idx] += 1
    for node_idx in range(num_nodes):
        if cnt[node_idx] != 1:
            remain_nodes.append(node_idx)
        if cnt[node_idx] > 1:
            for client_id in range(num_clients):
                if node_idx in node_dict[client_id]:
                    node_dict[client_id].remove(node_idx)
    num_remain_nodes = len(remain_nodes)
    remain_node_dict = dirichlet_distribution(num_entity=num_remain_nodes, num_assignments=num_clients)
    for client_id in range(num_clients):
        node_dict[client_id] = list(node_dict[client_id])
        for remain_idx in remain_node_dict[client_id]:
            node_dict[client_id].append(remain_nodes[remain_idx])
    return node_dict

def structure_non_iid_ego(graph, num_clients):
    num_nodes = graph.number_of_nodes()
    ego_dict = dirichlet_distribution(num_entity=num_nodes, num_assignments=num_clients)
    node_dict = {client_id:set() for client_id in range(num_clients)}
    A = nx.to_scipy_sparse_matrix(graph)
    ego_graphs = [set([node_idx] + A[node_idx].nonzero()[1].tolist())  for node_idx in range(num_nodes)]
    for client_id in range(num_clients):
        for ego_idx in ego_dict[client_id]:
            node_dict[client_id] |= ego_graphs[ego_idx]
    cnt = [0] * num_nodes
    remain_nodes = []
    for client_id in range(num_clients):
        for node_idx in node_dict[client_id]:
            cnt[node_idx] += 1
    for node_idx in range(num_nodes):
        if cnt[node_idx] > 1:
            remain_nodes.append(node_idx)
            for client_id in range(num_clients):
                if node_idx in node_dict[client_id]:
                    node_dict[client_id].remove(node_idx)
    num_remain_nodes = len(remain_nodes)
    remain_node_dict = dirichlet_distribution(num_entity=num_remain_nodes, num_assignments=num_clients)
    for client_id in range(num_clients):
        node_dict[client_id] = list(node_dict[client_id])
        for remain_idx in remain_node_dict[client_id]:
            node_dict[client_id].append(remain_nodes[remain_idx])
    return node_dict