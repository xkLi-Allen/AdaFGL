import torch
import numpy as np
import os
import os.path as osp
import scipy.sparse as sp
from torch import Tensor
from scipy.sparse import csr_matrix
from torch_geometric.data import Data


def load_geom_data(root, name):
    processed_dir = osp.join('./', root, name, 'processed')
    processed_file = osp.join(processed_dir, 'data.pt')
    try:
        return torch.load(processed_file)
    except:
        if not osp.exists(processed_dir):
            os.makedirs(processed_dir)
        raw_dir = osp.join(root, name, 'raw')
        f = np.loadtxt(raw_dir + '/{}.feature'.format(name.lower()), dtype=float)
        l = np.loadtxt(raw_dir + '/{}.label'.format(name.lower()), dtype=int)
        x = sp.csr_matrix(f, dtype=np.float32).tolil()
        x = normalize(x)
        x = torch.FloatTensor(np.array(x.todense()))
        y = torch.LongTensor(np.array(l))
        struct_edges = np.genfromtxt(raw_dir + '/{}.edge'.format(name.lower()), dtype=np.int32)
        edge_index = list(struct_edges)
        for i in range(len(struct_edges)):
            edge_index.append((struct_edges[i][1], struct_edges[i][0]))
        sedges = np.array(edge_index, dtype=np.int32)
        edge_idx = torch.tensor(sedges, dtype=torch.long).T
        data = Data(x=x,
                    y=y,
                    edge_index=edge_idx)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        data.input_dim = data.num_features  
        data.output_dim = data.y.max().item() + 1
        data.adj = sp.coo_matrix((torch.ones([len(data.edge_index[0])]), (data.edge_index[0], data.edge_index[1])), shape=(data.num_nodes, data.num_nodes))
        data.row, data.col, data.edge_weight = data.adj.row, data.adj.col, data.adj.data
        if isinstance(data.row, Tensor) or isinstance(data.col, Tensor):
            data.adj = csr_matrix((data.edge_weight.numpy(), (data.row.numpy(), data.col.numpy())),
                                            shape=(data.num_nodes, data.num_nodes))
        else:
            data.adj = csr_matrix((data.edge_weight, (data.row, data. col)), shape=(data.num_nodes, data.num_nodes))
        torch.save(data, processed_file)
        return data

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx