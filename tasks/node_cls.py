import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tasks.base_task import BaseTask
from tasks.utils import train, evaluate, accuracy



class SGLNodeClassification(BaseTask):
    def __init__(self, dataset, model, lr, weight_decay, epochs, device, show_epoch_info = 20, loss_fn=nn.CrossEntropyLoss()):
        super(SGLNodeClassification, self).__init__()
        self.dataset = dataset
        self.labels = self.dataset.y
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.epochs = epochs
        self.show_epoch_info = show_epoch_info
        self.loss_fn = loss_fn
        self.device = device

    def execute(self):
        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)
        best_val = 0.
        best_test = 0.
        for epoch in range(self.epochs):
            t = time.time()
            loss_train, acc_train = train(self.model, self.dataset.train_idx, self.labels, self.device,
                                            self.optimizer, self.loss_fn)
            acc_val, acc_test = evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                            self.labels, self.device)
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
        acc_val, acc_test = self.postprocess()
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test
        return best_val, best_test, self.model
    
    def postprocess(self):
        self.model.eval()
        outputs = self.model.model_forward(
            range(self.dataset.num_nodes), self.device).to("cpu")
        final_output = self.model.postprocess(self.dataset.adj, outputs)
        acc_val = accuracy(
            final_output[self.dataset.val_idx], self.labels[self.dataset.val_idx])
        acc_test = accuracy(
            final_output[self.dataset.test_idx], self.labels[self.dataset.test_idx])
        return acc_val, acc_test


class SGLEvaluateModelClients(BaseTask):
    def __init__(self, dataset, model, device):
        super(SGLEvaluateModelClients, self).__init__()
        self.dataset = dataset
        self.labels = self.dataset.y
        self.model = model
        self.device = device

    def execute(self):
        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)
        acc_val, acc_test = evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                        self.labels, self.device)
        acc_val, acc_test = self.postprocess()
        return acc_val, acc_test

    def postprocess(self):
        self.model.eval()
        outputs = self.model.model_forward(
            range(self.dataset.num_nodes), self.device).to("cpu")
        final_output = self.model.postprocess(self.dataset.adj, outputs)
        acc_val = accuracy(
            final_output[self.dataset.val_idx], self.labels[self.dataset.val_idx])
        acc_test = accuracy(
            final_output[self.dataset.test_idx], self.labels[self.dataset.test_idx])
        return acc_val, acc_test

