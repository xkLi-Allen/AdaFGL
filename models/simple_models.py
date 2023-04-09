import torch.nn as nn



class IdenticalMapping(nn.Module):
    def __init__(self) -> None:
        super(IdenticalMapping, self).__init__()

    def forward(self, feature):
        return feature


class LogisticRegression(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(feat_dim, output_dim)

    def forward(self, feature):
        output = self.fc(feature)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.5, bn=False, ln=False):
        super(MultiLayerPerceptron, self).__init__()
        
        if num_layers < 2:
            raise ValueError("MLP must have at least two layers!")
        self.num_layers = num_layers

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        self.ln = ln

        self.norms = nn.ModuleList()
        if bn:
            for _ in range(num_layers-1):
                self.norms.append(nn.BatchNorm1d(hidden_dim))
        if ln:
            for _ in range(num_layers-1):
                self.norms.append(nn.LayerNorm(hidden_dim))


        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            nn.init.zeros_(fc.bias)

    def forward(self, feature):
        for i in range(self.num_layers - 1):
            feature = self.fcs[i](feature)
            if self.bn is True or self.ln is True:
                feature = self.norms[i](feature)
            feature = self.prelu(feature)
            feature = self.dropout(feature)

        output = self.fcs[-1](feature)
        return output


class ResMultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout=0.8, bn=False, ln=False):
        super(ResMultiLayerPerceptron, self).__init__()
        if num_layers < 2:
            raise ValueError("ResMLP must have at least two layers!")
        self.num_layers = num_layers

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        self.ln = ln

        self.norms = nn.ModuleList()
        if bn:
            for _ in range(num_layers-1):
                self.norms.append(nn.BatchNorm1d(hidden_dim))
        if ln:
            for _ in range(num_layers-1):
                self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, feature):
        feature = self.dropout(feature)
        feature = self.fcs[0](feature)
        if self.bn is True or self.ln is True:
            feature = self.norms[0](feature)
        feature = self.relu(feature)
        residual = feature

        for i in range(1, self.num_layers - 1):
            feature = self.dropout(feature)
            feature = self.fcs[i](feature)
            if self.bn is True or self.ln is True:
                feature = self.norms[i](feature)
            feature_ = self.relu(feature)
            feature = feature_ + residual
            residual = feature_

        feature = self.dropout(feature)
        output = self.fcs[-1](feature)
        return output