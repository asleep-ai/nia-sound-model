import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from dataset import SleepSoundDataset
from model import NeuralNetwork
from utils import set_seed, save_model, load_model


def draw_fig(data, title, y_label, save_path):
    plt.figure()
    plt.plot(data, c='b')
    plt.title(title)
    plt.xlabel('Training epochs')
    plt.ylabel(y_label)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title}.png', dpi=300)
    plt.close()


def train(dataloader, model, loss_fn, optimizer):
    model.train()
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


def inference(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    all_pred, all_label = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            pred = torch.argmax(pred, dim=1)
            all_pred += pred.tolist()
            all_label += y.tolist()
    loss /= num_batches

    acc = (np.array(all_label) == np.array(all_pred)).mean()
    return acc, loss, all_pred, all_label


def get_prob_list(train_loader, model, prob_file):
    model.eval()
    train_osarisk_probs = []
    with torch.no_grad():
        for X, _ in train_loader:
            X = X.to(device)
            logits = model(X)
            probs = softmax(logits, dim=1)
            train_osarisk_probs += probs[:, 1].tolist()  # Probs of having OSA

    train_osarisk_probs = np.array(train_osarisk_probs)
    np.save(prob_file, train_osarisk_probs)

    return train_osarisk_probs


if __name__ == '__main__':
    set_seed(seed=0)

    # Save path
    save_path = 'checkpoint'
    os.makedirs(save_path, exist_ok=False)

    batch_size = 8
    train_npy = 'data/train.npy'
    valid_npy = 'data/valid.npy'

    # Create the dataset for train, validation
    training_data = SleepSoundDataset(npy_file=train_npy)
    valid_data = SleepSoundDataset(npy_file=valid_npy)

    # Create a DataLoader for the training and valid data
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size)

    # Get resources to run the experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Create the model
    model = NeuralNetwork().to(device)

    # Create the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 500
    best_acc = -1.
    all_train_loss = []
    all_valid_loss = []
    all_valid_acc = []
    for epoch in range(num_epochs):
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        valid_acc, valid_loss, all_pred, all_label = inference(valid_dataloader, model, loss_fn)
        all_train_loss.append(train_loss)
        all_valid_loss.append(valid_loss)
        all_valid_acc.append(valid_acc)
        if epoch % 10 == 0:
            print(f'Epoch {epoch} -------------------------------')
            print(f'[Train] Loss: {train_loss:.4f}')
            print(f'[Valid] Avg loss: {valid_loss:.4f} Acc: {valid_acc:.2%}')
            print(classification_report(all_label, all_pred,
                                        target_names=['No risk', 'OSA risk'], zero_division=0))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            best_pred = all_pred
            best_label = all_label
            save_model(model, os.path.join(save_path, 'best_model.pt'))

    print('------------------------------------------')
    print(f'Highest validation accuracy {best_acc:.2%} achieved at epoch {best_epoch}:')
    print(classification_report(best_label, best_pred,
                                target_names=['No risk', 'OSA risk'], zero_division=0))

    print('\nFigure drawing...')
    draw_fig(np.array(all_valid_acc) * 100, 'Validation accuracy', 'Accuracy (%)', save_path)
    draw_fig(all_train_loss, 'Training loss', 'Loss value', save_path)
    draw_fig(all_valid_loss, 'Validation loss', 'Loss value', save_path)

    # Saving probabilities of having OSA on training set
    print('\nSaving probabilities of having OSA from training set...')
    model_pt_file = os.path.join(save_path, 'best_model.pt')
    load_model(model, model_file=model_pt_file)
    prob_file = os.path.join(save_path, 'train_osarisk_probs.npy')
    get_prob_list(train_dataloader, model, prob_file)

    print('Done!')
