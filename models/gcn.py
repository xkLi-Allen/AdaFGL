import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_mx_to_torch_sparse_tensor, adj_to_symmetric_norm



class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_hat):
        x = self.linear(x)
        x = torch.mm(adjacency_hat, x)
        return x


class ChebNetConv(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(ChebNetConv, self).__init__()

        self.K = k
        self.linear = nn.Linear(in_features * k, out_features)

    def forward(self, x, laplacian):
        x = self.__transform_to_chebyshev(x, laplacian)
        x = self.linear(x)
        return x

    def __transform_to_chebyshev(self, x, laplacian):
        cheb_x = x.unsqueeze(2)
        x0 = x

        if self.K > 1:
            x1 = torch.mm(laplacian, x0)
            cheb_x = torch.cat((cheb_x, x1.unsqueeze(2)), 2)
            for _ in range(2, self.K):
                x2 = 2 * torch.mm(laplacian, x1) - x0
                cheb_x = torch.cat((cheb_x, x2.unsqueeze(2)), 2)
                x0, x1 = x1, x2

        cheb_x = cheb_x.reshape([x.shape[0], -1])
        return cheb_x



class GCN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5, bn=False, ln=False):
        super(GCN, self).__init__()

        self.use_graph_op = True
        self.pre_graph_op = None

        self.conv1 = GCNConv(feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.post_graph_op = None

    def preprocess(self, adj, feature):
        self.pre_msg_learnable = False
        self.processed_feature = feature

        self.adj = adj_to_symmetric_norm(adj, r=0.5)

        self.adj = sparse_mx_to_torch_sparse_tensor(self.adj)

    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        processed_feature = None
        if self.pre_msg_learnable is False:
            processed_feature = self.processed_feature.to(device)
        else:
            transferred_feat_list = [feat.to(
                device) for feat in self.processed_feat_list]
            processed_feature = self.pre_msg_op.aggregate(
                transferred_feat_list)

        self.adj = self.adj.to(device)
        x = processed_feature
        x = self.conv1(x, self.adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.adj)

        return x[idx]

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)
        return output


class ChebNet(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.1, bn=False, ln=False, k=2):
        super(ChebNet, self).__init__()

        self.use_graph_op = True
        self.pre_graph_op = None

        self.conv1 = ChebNetConv(feat_dim, hidden_dim, k)
        self.conv2 = ChebNetConv(hidden_dim, output_dim, k)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.post_graph_op = None

    def preprocess(self, adj, feature):
        self.pre_msg_learnable = False
        self.processed_feature = feature
        
        adj = adj_to_symmetric_norm(adj, r=0.5)

        self.adj = sparse_mx_to_torch_sparse_tensor(adj)

    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        processed_feature = None
        if self.pre_msg_learnable is False:
            processed_feature = self.processed_feature.to(device)
        else:
            transferred_feat_list = [feat.to(
                device) for feat in self.processed_feat_list]
            processed_feature = self.pre_msg_op.aggregate(
                transferred_feat_list)

        self.adj = self.adj.to(device)
        x = processed_feature
        x = self.conv1(x, self.adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.adj)

        return x[idx]

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)
        return output