import copy
import time
import numpy as np
from config import args
from models.sgl_models import SGC, MLP, SGCMLP
from models.gcn import GCN, ChebNet
from models.gat import GAT
from tasks.node_cls import SGLNodeClassification

class ClientsManager():
    def __init__(self, model_name, datasets, num_clients, device, eval_single_client=False):
        self.hidden_dim = args.hidden_dim
        self.model_name = model_name
        self.global_data = datasets.global_data
        self.input_dim = datasets.input_dim
        self.output_dim = datasets.output_dim
        self.subgraphs = datasets.subgraphs
        self.device = device
        self.num_clients = num_clients
        self.clients = []
        self.tot_nodes = 0
        self.initClient()
        if eval_single_client:
            self.evaluate_data_isolate()

    def initClient(self):
        for client_id in range(self.num_clients):
            client = Client(
                model_name = self.model_name, 
                input_dim = self.input_dim, 
                output_dim = self.output_dim, 
                client_id = client_id, 
                local_subgraph = self.subgraphs[client_id],
                hidden_dim = self.hidden_dim
            )
            self.clients.append(client)
            self.tot_nodes += client.num_nodes

class Client(object):
    def __init__(self, model_name, input_dim, output_dim, client_id, local_subgraph, hidden_dim):
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.client_id = client_id
        self.local_subgraph = local_subgraph
        self.model_records = []
        self.num_nodes = self.local_subgraph.num_nodes
        self.hidden_dim = hidden_dim
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
            self.model = GAT(feat_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim, dropout=args.drop)
        elif self.model_name == "ChebNet":
            self.model = ChebNet(feat_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim, dropout=args.drop, bn=False, ln=False)

    def clear_record(self):
        self.model_records = []

    def add_record(self, model):
        self.model_records.append(copy.deepcopy(model))

    def set_state_dict(self, model):
        self.model.load_state_dict(state_dict=copy.deepcopy(model.state_dict()))








