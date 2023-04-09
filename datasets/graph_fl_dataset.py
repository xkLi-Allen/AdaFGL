import time
import torch
import os
import os.path as osp
import scipy.sparse as sp
from torch import Tensor
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from datasets.data_sampling import data_partitioning
from datasets.geom_data import load_geom_data
from torch_geometric.data import Dataset
from datasets.utils import analysis_graph_structure_statis_info, analysis_graph_structure_homo_hete_info


class GraphFLDataset(Dataset):
    def __init__(self, root, name, sampling, num_clients,
        analysis_local_subgraph,
        analysis_global_graph,
        ratio_train = 0.2,
        ratio_val = 0.4,
        ratio_test = 0.4,
        ratio_iso = 0.5,
        ratio_homo = 0.001,
        ratio_hete = 0.001,
        transform = None, pre_transform = None, pre_filter = None): 
        self.name = name   
        self.sampling = sampling
        self.num_clients = num_clients
        self.ratio_train = ratio_train
        self.ratio_val = ratio_val
        self.ratio_test = ratio_test
        self.ratio_homo = ratio_homo
        self.ratio_hete = ratio_hete
        self.ratio_iso = ratio_iso
        self.analysis_local_subgraph = analysis_local_subgraph
        self.analysis_global_graph = analysis_global_graph
        super(GraphFLDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data()

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "Planetoid") if self.name in ["Cora", "CiteSeer", "PubMed"] else self.root

    @property
    def processed_dir(self) -> str:
        return osp.join(self.raw_dir, self.name, "Client{}".format(self.num_clients), self.sampling)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        files_names = ['data{}.pt'.format(i) for i in range(self.num_clients)]
        return files_names

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data{}.pt'.format(idx)))
        return data

    def load_global_graph(self, process=False):
        print("| ★  Load Global Data: {}".format(self.name))
        if self.name in ["Cora", "CiteSeer", "PubMed"]:
            self.global_dataset = Planetoid(root=self.raw_dir, name=self.name)     
            self.input_dim = self.global_dataset.num_features  
            self.output_dim = self.global_dataset.num_classes
            self.global_data = self.global_dataset.data
            self.global_data.adj = sp.coo_matrix((torch.ones([len(self.global_data.edge_index[0])]), (self.global_data.edge_index[0], self.global_data.edge_index[1])), shape=(self.global_data.num_nodes, self.global_data.num_nodes))
            self.global_data.row, self.global_data.col, self.global_data.edge_weight = self.global_data.adj.row, self.global_data.adj.col, self.global_data.adj.data
            if isinstance(self.global_data.row, Tensor) or isinstance(self.global_data.col, Tensor):
                self.global_data.adj = csr_matrix((self.global_data.edge_weight.numpy(), (self.global_data.row.numpy(), self.global_data.col.numpy())),
                                                shape=(self.global_data.num_nodes, self.global_data.num_nodes))
            else:
                self.global_data.adj = csr_matrix((self.global_data.edge_weight, (self.global_data.row, self.global_data. col)), shape=(self.global_data.num_nodes, self.global_data.num_nodes))
        elif self.name in ["Chameleon", "Squirrel"]:
            self.global_data = load_geom_data(root=self.raw_dir, name=self.name)
            self.input_dim = self.global_data.input_dim  
            self.output_dim = self.global_data.output_dim
        else:
            raise ValueError("Not supported for this dataset, please check root file path and dataset name")
                        
    def process(self):
        self.load_global_graph(process=True)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        subgraph_list = data_partitioning(
            G=self.global_data,
            sampling=self.sampling,
            num_clients=self.num_clients,
            ratio_train=self.ratio_train,
            ratio_val=self.ratio_val,
            ratio_test=self.ratio_test,
            ratio_iso=self.ratio_iso,
            ratio_homo=self.ratio_homo,
            ratio_hete=self.ratio_hete
        )
        for i in range(self.num_clients):
            torch.save(subgraph_list[i], self.processed_paths[i])

    def load_data(self):
        self.load_global_graph()
        self.subgraphs = [self.get(i) for i in range(self.num_clients)]
        for i in range(len(self.subgraphs)):
            if i == 0:
                self.global_data.train_idx = self.subgraphs[i].global_train_idx
                self.global_data.val_idx = self.subgraphs[i].global_val_idx
                self.global_data.test_idx = self.subgraphs[i].global_test_idx
            else:
                self.global_data.train_idx += self.subgraphs[i].global_train_idx
                self.global_data.val_idx += self.subgraphs[i].global_val_idx
                self.global_data.test_idx += self.subgraphs[i].global_test_idx
        if self.analysis_local_subgraph:
            for i in range(len(self.subgraphs)):
                structure_statis_info = analysis_graph_structure_statis_info(self.subgraphs[i])
                structure_homo_hete_label_info, structure_homo_hete_feature_info = analysis_graph_structure_homo_hete_info(self.subgraphs[i])
