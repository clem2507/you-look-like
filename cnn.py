import torch

def train_loop(model, train_dataloader, device, optimizer, criterion):
    output = {'loss': 0.,
              'accuracy': 0.}
    
    train_epoch_loss = 0
    train_epoch_acc = 0

    train_running_loss = 0
    train_running_acc = 0

    size_dataset = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    for batch, data in enumerate(train_dataloader):
        X = data[0].to(device)
        y_true = data[1].to(device)

        # Compute prediction, loss and accuracy
        y_model = model(X)
        train_loss = criterion(y_model, y_true)
        train_acc = multi_acc(y_model, y_true)

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Cumulate loss and accuracy (for every 50 batchs)
        train_running_loss += train_loss.item()

        # Cumulate loss and accuracy (for all batchs of the epoch)
        train_epoch_loss += train_loss.item()

        # Display loss and accuracy every 50 batchs
        if batch % 50 == 0:
            a = y_true.int().cpu().numpy()
            b = y_model.detach().cpu().numpy()
            print(f"[{batch:>5d}/{num_batches:>5d}] train_loss: {train_running_loss / 50:.3f}, accuracy: {train_acc:.3f} ")
            train_running_loss = 0

    output['loss'] = train_epoch_loss
    output['accuracy'] = train_acc

    return output['loss'], output['accuracy']



def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    output = {'valid_loss': 0.,
              'running_loss': 0.}

    size_dataset = len(valid_dataloader.dataset)
    num_batches = len(valid_dataloader)

    valid_loss = 0
    running_loss = 0
    
    with torch.no_grad():
        for data in valid_dataloader:
            X = data[0].to(device)
            y_true = data[1].to(device)

            # Compute prediction and loss
            y_model = model(X)
            loss = criterion(y_model, y_true)
            valid_loss += loss.item()
            running_loss += loss.item()

    output['valid_loss'] = valid_loss
    output['running_loss'] = running_loss / 50

    return output['valid_loss'], output['running_loss']


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc