import random
import torch
import torch.nn.functional as F
from datasets.utils import idx_to_mask
from models.message_op.laplacian_graph_op import LaplacianGraphOp


class NonParaLP():
    def __init__(self, prop_steps, num_class, alpha, r=0.5):
        self.prop_steps = prop_steps
        self.r = r
        self.num_class = num_class
        self.alpha = alpha

        self.graph_op = LaplacianGraphOp(prop_steps=self.prop_steps, r=self.r)

    def preprocess(self, nodes_embedding, subgraph, device):
        self.subgraph = subgraph
        self.y = subgraph.y
        self.label = F.one_hot(self.y.view(-1), self.num_class).to(torch.float).to(device)
        num_nodes = len(self.subgraph.train_idx)

        train_idx_list = torch.where(self.subgraph.train_idx == True)[0].numpy().tolist()
        num_train = int(len(train_idx_list) / 2)

        random.shuffle(train_idx_list)
        self.lp_train_idx = idx_to_mask(train_idx_list[: num_train], num_nodes)
        self.lp_eval_idx = idx_to_mask(train_idx_list[num_train: ], num_nodes)

        unlabel_idx = self.lp_eval_idx | self.subgraph.val_idx | self.subgraph.test_idx
        unlabel_init = torch.full([self.label[unlabel_idx].shape[0], self.label[unlabel_idx].shape[1]], 1 / self.num_class).to(device)
        self.label[self.lp_eval_idx + self.subgraph.val_idx + self.subgraph.test_idx] = unlabel_init
        
    def propagate(self, adj):
        self.output = self.graph_op.init_lp_propagate(adj, self.label, init_label=self.lp_train_idx, alpha=self.alpha)
        self.output = self.output[-1]

    def eval(self, i=None):
        pred = self.output.max(1)[1].type_as(self.subgraph.y)
        correct = pred[self.lp_eval_idx].eq(self.subgraph.y[self.lp_eval_idx]).double()
        correct = correct.sum()
        reliability_acc = (correct / self.subgraph.y[self.lp_eval_idx].shape[0]).item()
        return reliability_acc
