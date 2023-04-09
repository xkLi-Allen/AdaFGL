import time
import numpy as np
import random
import torch
import os.path as osp
from config import args
from collections import OrderedDict
from models.sgl_models import SGC, MLP, SGCMLP
from models.gat import GAT
from models.gcn import GCN, ChebNet
from tasks.node_cls import SGLNodeClassification, SGLEvaluateModelClients



class ServerManager():
    def __init__(self, model_name, datasets, num_clients, device, num_rounds, client_sample_ratio):
        self.hidden_dim = args.hidden_dim
        self.model_name = model_name
        self.datasets = datasets
        self.input_dim = datasets.input_dim
        self.output_dim = datasets.output_dim
        self.global_data = datasets.global_data
        self.subgraphs = datasets.subgraphs
        self.num_clients = num_clients
        self.device = device
        self.client_sample_ratio = client_sample_ratio
        self.state_dict_records = []
        self.num_rounds = num_rounds
        self.init_model()

    def init_model(self):
        if self.model_name == "SGC":
            self.model = SGC(prop_steps=3, feat_dim=self.input_dim, output_dim=self.output_dim)
        if self.model_name == "SGCMLP":
            self.model = SGCMLP(prop_steps=3, feat_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=3, output_dim=self.output_dim, dropout=args.drop, bn=False, ln=False)
        elif self.model_name == "MLP":
            self.model = MLP(feat_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=3, output_dim=self.output_dim, dropout=args.drop, bn=False, ln=False)
        elif self.model_name == "GCN":
            self.model = GCN(feat_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim, dropout=args.drop, bn=False, ln=False)
        elif self.model_name == "GAT":
            self.model = GAT(feat_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim,
                             dropout=args.drop)
        elif self.model_name == "ChebNet":
            self.model = ChebNet(feat_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim, dropout=args.drop, bn=False, ln=False)

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict=state_dict)

    def model_aggregation(self, server_input, mixing_coefficients):
        aggregated_model = OrderedDict()

        server_input = [a.state_dict() for a in server_input]
        for it, state_dict in enumerate(server_input):
            for key in state_dict.keys():
                if it == 0:
                    aggregated_model[key] = mixing_coefficients[it] * state_dict[key]
                else:
                    aggregated_model[key] += mixing_coefficients[it] * state_dict[key]
        return aggregated_model

    def collaborative_training_model(self, clients, data_name, num_clients, sampling, model_name, normalize_trains=args.normalize_train, lr=args.lr, weight_decay=args.weight_decay, epochs=args.num_epochs):
        print("| ★  Start Training Federated GNN Model...")
        normalize_record = {"val_acc": [], "test_acc": []}
        t_total = time.time()
        for _ in range(normalize_trains):
            clients_test_acc = []
            clients_val_acc = []
            self.init_model()
            for client_id in range(self.num_clients):
                clients_test_acc.append(0)
                clients_val_acc.append(0)
                clients[client_id].clear_record()
                clients[client_id].init_model()
                clients[client_id].model.preprocess(clients[client_id].local_subgraph.adj, clients[client_id].local_subgraph.x)
            round_global_record = {"global_val_acc": 0, "global_test_acc": 0}
            for round_id in range(self.num_rounds):
                all_client_idx = list(range(self.num_clients))
                random.shuffle(all_client_idx)
                sample_num = int(len(all_client_idx) * self.client_sample_ratio)
                sample_idx = sorted(all_client_idx[:sample_num])
                mixing_coefficients = [clients[idx].num_nodes for idx in sample_idx]
                mixing_coefficients = [val / sum(mixing_coefficients) for val in mixing_coefficients]
                aggregated_model_list = []
                for client_id in sample_idx:
                    clients[client_id].set_state_dict(self.model)
                    _, _, local_model = SGLNodeClassification(dataset = clients[client_id].local_subgraph, 
                    model = clients[client_id].model, 
                    lr = lr, 
                    weight_decay = weight_decay, 
                    epochs = epochs, 
                    device = self.device).execute()
                    aggregated_model_list.append(local_model)
                aggregated_model = self.model_aggregation(aggregated_model_list, mixing_coefficients)
                self.set_state_dict(aggregated_model)
                global_val_acc = 0
                global_test_acc = 0
                for client_id in range(self.num_clients):
                    self.model.pre_msg_learnable = clients[client_id].model.pre_msg_learnable
                    self.model.processed_feature = clients[client_id].model.processed_feature
                    self.model.adj = clients[client_id].model.adj
                    val_acc, test_acc = SGLEvaluateModelClients(dataset = clients[client_id].local_subgraph, 
                    model = self.model, 
                    device = self.device).execute()
                    if val_acc > clients_val_acc[client_id]:
                        clients_val_acc[client_id] = val_acc
                        clients_test_acc[client_id] = test_acc
                    global_val_acc += (val_acc * clients[client_id].local_subgraph.num_nodes / self.datasets.global_data.num_nodes)
                    global_test_acc += (test_acc * clients[client_id].local_subgraph.num_nodes / self.datasets.global_data.num_nodes)
                if global_val_acc > round_global_record["global_val_acc"]:
                    round_global_record["global_val_acc"] = global_val_acc
                    round_global_record["global_test_acc"] = global_test_acc
                    if normalize_trains == 1:
                        torch.save(self.model, osp.join("./model_weights", "{}_Client{}_{}_{}.pt".format(data_name, num_clients, sampling, model_name)))
            normalize_record["val_acc"].append(round_global_record["global_val_acc"])
            normalize_record["test_acc"].append(round_global_record["global_test_acc"])
