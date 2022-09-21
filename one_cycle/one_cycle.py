import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

import dataset


def set_seed(seed):
    """Set random seeds for random, numpy, torch, and torch.backends.cudnn."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def save_model(model, model_path):
    model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
    torch.save(model_state_dict, model_path)

def draw_fig(data, title, y_label):
    plt.figure()
    plt.plot(data, c='b')
    plt.title(title)
    plt.xlabel('Training epochs')
    plt.ylabel(y_label)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.close()


class NeuralNetwork(nn.Module):
    """Define the neural network model."""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        model_size = 32
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40, model_size),
            nn.ReLU(),
            nn.Linear(model_size, 2)
        )

    def forward(self, x):
        """The forward pass of the model."""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    for (X, y) in dataloader:
        X, y = X.to(device), y.to(device)

        # forward pass
        pred = model(X)

        # compute loss
        loss = loss_fn(pred, y)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    all_pred, all_label = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = torch.argmax(pred, dim=1)
            all_pred += pred.tolist()
            all_label += y.tolist()
    test_loss /= num_batches

    acc = (np.array(all_label) == np.array(all_pred)).mean()
    return acc, test_loss, all_pred, all_label


if __name__ == '__main__':
    set_seed(seed=0)

    batch_size = 8
    root = '/HDD/nia/data'

    # Create the dataset for training and testing
    training_data = dataset.SleepSoundDataset(root=root, train=True)
    test_data = dataset.SleepSoundDataset(root=root, train=False)

    # Create a DataLoader for the training and test data
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get resources to run the experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Create the model
    model = NeuralNetwork().to(device)

    # Create the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 100
    best_acc = -1.
    all_train_loss = []
    all_test_loss = []
    all_acc = []
    for epoch in range(num_epochs):
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        acc, test_loss, all_pred, all_label = test(test_dataloader, model, loss_fn)
        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)
        all_acc.append(acc)
        if epoch % 10 == 0:
            print(f'Epoch {epoch} -------------------------------')
            print(f'[Train] Loss: {train_loss:.4f}')
            print(f'[Test ] Avg loss: {test_loss:.4f} Acc: {acc:.2%}')
            print(classification_report(all_label, all_pred,
                                        target_names=['No risk', 'OSA risk'], zero_division=0))
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_pred = all_pred
            best_label = all_label
            save_model(model, 'best_model.pt')

    print('------------------------------------------')
    print(f'Highest accuracy {best_acc:.2%} achieved at epoch {best_epoch}:')
    print(classification_report(best_label, best_pred,
                                target_names=['No risk', 'OSA risk'], zero_division=0))

    print('\nFigure drawing...')
    draw_fig(np.array(all_acc) * 100, title='Test accuracy', y_label='Accuracy (%)')
    draw_fig(all_train_loss, title='Training loss', y_label='Loss value')
    draw_fig(all_test_loss, title='Test loss', y_label='Loss value')

    print('Done!')
