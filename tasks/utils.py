def train(model, train_idx, labels, device, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    train_output = model.model_forward(train_idx, device)
    loss_train = loss_fn(train_output, labels[train_idx])
    acc_train = accuracy(train_output, labels[train_idx])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train


def evaluate(model, val_idx, test_idx, labels, device):
    model.eval()
    output = model.model_forward(range(len(val_idx)), device)
    acc_val = accuracy(output[val_idx], labels[val_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    return acc_val, acc_test

    
def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)).item()
