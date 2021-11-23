import torch


def test(dataloader, model, loss_function, device):

    cum_loss, correct_pred = 0, 0
    for X, y in dataloader:

        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            # Compute prediction, loss and correct predictions
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()

            cum_loss += loss
            correct_pred += correct

    average_loss = cum_loss / len(dataloader)
    average_accuracy = correct_pred / len(dataloader.dataset)

    # print('Test error: {}\nAccuracy {}\n'.format(average_loss, average_accuracy))

    return average_loss, average_accuracy