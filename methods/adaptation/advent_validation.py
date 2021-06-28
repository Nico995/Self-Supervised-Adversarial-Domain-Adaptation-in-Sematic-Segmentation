import torch

from utils import reverse_one_hot, global_accuracy, get_confusion_matrix


def advent_validation(model, data, label, criterion, loss, classes):
    # Disable dropout and batch norm layers
    model.eval()

    # Don't compute gradients during evaluation step
    with torch.no_grad():
        # get model output and remove batch dimension
        predict = model(data).squeeze()
        # Get the prediction by getting the maximum probability along dimension 0
        predict = reverse_one_hot(predict)
        # predict = np.array(predict.detach().cpu())

        # get RGB label image and remove batch dimension
        label = label.squeeze()
        if loss == 'dice':
            label = reverse_one_hot(label)

        # label = np.array(label.detach().cpu())

        # compute per pixel accuracy
        precision = global_accuracy(predict, label)
        confusion_matrix = get_confusion_matrix(predict.flatten(), label.flatten(), classes)

        return precision, confusion_matrix
