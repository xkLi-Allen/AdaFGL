import torch.nn as nn
import torch.nn.functional as F
from models.message_op.laplacian_graph_op import LaplacianGraphOp
from models.message_op.last_message_op import LastMessageOp
from models.simple_models import LogisticRegression, MultiLayerPerceptron
from utils import sparse_mx_to_torch_sparse_tensor, adj_to_symmetric_norm

class SGC(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim, r=0.5):
        super(SGC, self).__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.r = r
        
        self.use_graph_op = True
        self.pre_graph_op = LaplacianGraphOp(prop_steps=self.prop_steps, r=self.r)
        self.pre_msg_op = LastMessageOp()
        self.base_model = LogisticRegression(feat_dim=self.feat_dim, output_dim=self.output_dim)
        self.post_graph_op = None
        self.post_msg_op = None

    def preprocess(self, adj, feature):
        self.processed_feat_list = self.pre_graph_op.propagate(
            adj, feature)
        if self.pre_msg_op._aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
            self.pre_msg_learnable = True
        else:
            self.pre_msg_learnable = False
            self.processed_feature = self.pre_msg_op.aggregate(self.processed_feat_list)
        self.adj = self.pre_graph_op.adj
        
    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op._aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output


    # a wrapper of the forward function
    def model_forward(self, idx, device):
        return self.forward(idx, device)


    def forward(self, idx, device):
        processed_feature = None
        if self.pre_msg_learnable is False:
            processed_feature = self.processed_feature[idx].to(device)
        else:
            transferred_feat_list = [feat[idx].to(
                device) for feat in self.processed_feat_list]
            processed_feature = self.pre_msg_op.aggregate(
                transferred_feat_list)

        output = self.base_model(processed_feature)
        return output

class SGCMLP(nn.Module):
    def __init__(self, prop_steps, num_layers, feat_dim, hidden_dim, output_dim, r=0.5, dropout=0.5, bn=False, ln=False):
        super(SGCMLP, self).__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.r = r
        self.bn = bn
        self.ln = ln
        
        self.use_graph_op = True
        self.pre_graph_op = LaplacianGraphOp(prop_steps=self.prop_steps, r=self.r)
        self.pre_msg_op = LastMessageOp()
        self.base_model = MultiLayerPerceptron(
            feat_dim=self.feat_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers, 
            output_dim=self.output_dim, 
            dropout=self.dropout, 
            bn=self.bn,
            ln=self.ln)
        self.post_graph_op = None
        self.post_msg_op = None

    def preprocess(self, adj, feature):

        self.processed_feat_list = self.pre_graph_op.propagate(
            adj, feature)
        if self.pre_msg_op._aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
            self.pre_msg_learnable = True
        else:
            self.pre_msg_learnable = False
            self.processed_feature = self.pre_msg_op.aggregate(self.processed_feat_list)
        self.adj = self.pre_graph_op.adj

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op._aggr_type in ["proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output


    # a wrapper of the forward function
    def model_forward(self, idx, device):
        return self.forward(idx, device)


    def forward(self, idx, device):
        processed_feature = None
        if self.pre_msg_learnable is False:
            processed_feature = self.processed_feature[idx].to(device)
        else:
            transferred_feat_list = [feat[idx].to(
                device) for feat in self.processed_feat_list]
            processed_feature = self.pre_msg_op.aggregate(
                transferred_feat_list)

        output = self.base_model(processed_feature)
        return output


class MLP(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.5, bn=False, ln=False):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bn = bn
        self.ln = ln
        
        self.use_graph_op = False
        self.pre_graph_op = None

        self.base_model = MultiLayerPerceptron(
            feat_dim=self.feat_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers, 
            output_dim=self.output_dim, 
            dropout=self.dropout, 
            bn=self.bn,
            ln=self.ln)

        self.post_graph_op = None
        self.post_msg_op = None

    def preprocess(self, adj, feature):
        self.pre_msg_learnable = False
        self.processed_feature = feature


        self.adj = sparse_mx_to_torch_sparse_tensor(adj)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)

        return output


    # a wrapper of the forward function
    def model_forward(self, idx, device):
        return self.forward(idx, device)


    def forward(self, idx, device):
        processed_feature = None
        if self.pre_msg_learnable is False:
            processed_feature = self.processed_feature[idx].to(device)
        else:
            transferred_feat_list = [feat[idx].to(
                device) for feat in self.processed_feat_list]
            processed_feature = self.pre_msg_op.aggregate(
                transferred_feat_list)

        output = self.base_model(processed_feature)
        return output