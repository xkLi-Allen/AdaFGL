import random
import numpy as np
from random import choice
from config import args
from datasets.dirichlet_distribution import dirichlet_distribution



def Ratio_Homo(edge_index, label, ratio_homo, ratio_iso):
    ratio = {}
    ratio["homo_ratio"] = ratio_homo
    ratio["iso_ratio"] = ratio_iso
    num_local_nodes = len(label)
    local_node_idx = [i for i in range(num_local_nodes)]
    edge_index_np = np.array(edge_index).T
    for i in range(num_local_nodes):
        edge_u = local_node_idx[i]
        label_u = label[edge_u]
        for j in range(i + 1, num_local_nodes):
            edge_v = local_node_idx[j]
            label_v = label[edge_v]
            if label_u == label_v:
                edge_prob = np.random.choice([0,1], size=1, p=[1-ratio["homo_ratio"], ratio["homo_ratio"]])
                edge_prob_iso = 0
                if edge_index_np.size == 0 or edge_u not in edge_index_np[0]:
                    edge_prob_iso = np.random.choice([0,1], size=1, p=[1-ratio["iso_ratio"], ratio["iso_ratio"]])
                if edge_prob == 1 or edge_prob_iso == 1:
                    edge_index.append((edge_u, edge_v))
                    edge_index.append((edge_v, edge_u))
                    new_edge = np.array([(edge_u, edge_v), (edge_v, edge_u)]).T
                    if edge_index_np.size == 0:
                        edge_index_np = new_edge
                    else:
                        edge_index_np = np.hstack((edge_index_np, new_edge))
    return ratio, edge_index
 

def Ratio_Hete(edge_index, label, ratio_hete, ratio_iso):
    ratio = {}
    ratio["hete_ratio"] = ratio_hete
    ratio["iso_ratio"] = ratio_iso
    num_local_nodes = len(label)
    local_node_idx = [i for i in range(num_local_nodes)]
    edge_index_np = np.array(edge_index).T
    for i in range(num_local_nodes):
        edge_u = local_node_idx[i]
        label_u = label[edge_u]
        for j in range(i + 1, num_local_nodes):
            edge_v = local_node_idx[j]
            label_v = label[edge_v]
            if label_u != label_v:
                edge_prob = np.random.choice([0,1], size=1, p=[1-ratio["hete_ratio"], ratio["hete_ratio"]])
                edge_prob_iso = 0
                if edge_index_np.size == 0 or edge_u not in edge_index_np[0]:
                    edge_prob_iso = np.random.choice([0,1], size=1, p=[1-ratio["iso_ratio"], ratio["iso_ratio"]])
                if edge_prob == 1 or edge_prob_iso == 1:
                    edge_index.append((edge_u, edge_v))
                    edge_index.append((edge_v, edge_u))
                    new_edge = np.array([(edge_u, edge_v), (edge_v, edge_u)]).T     
                    if edge_index_np.size == 0:
                        edge_index_np = new_edge        
                    else:
                        edge_index_np = np.hstack((edge_index_np, new_edge))
    return ratio, edge_index

