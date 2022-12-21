import os

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from dataset import SleepSoundDataset
from model import NeuralNetwork
from utils import set_seed, load_model


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

def get_percentile(probs, train_osarisk_probs):
    percentiles = []
    for prob in probs:
        cnt_larger = (train_osarisk_probs >= prob).sum()
        percentile_val = (cnt_larger + 1) / len(train_osarisk_probs)
        percentiles.append(percentile_val)
    return percentiles

def nia_inference_api(model, batch_data, train_osarisk_probs):
    # Probabilities
    logits = model(batch_data)
    probs = softmax(logits, dim=1)
    probs = probs[:, 1].tolist()    # Probability of having OSA risk

    # Percentiles
    percentiles = get_percentile(probs, train_osarisk_probs)

    return probs, percentiles

def inference(test_loader, model, train_osarisk_probs):
    all_preds, all_labels, all_percentiles = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            probs, percentiles = nia_inference_api(model, X, train_osarisk_probs)
            preds = [1 if x >= 0.5 else 0 for x in probs]
            all_preds += preds
            all_labels += y.tolist()
            all_percentiles += percentiles
    return all_preds, all_labels, all_percentiles


if __name__ == '__main__':
    set_seed(seed=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Load model
    model = NeuralNetwork().to(device)
    model_pt_file = 'checkpoint/best_model.pt'
    load_model(model, model_file=model_pt_file)
    model.eval()

    batch_size = 8
    train_npy = 'data/train.npy'
    test_npy = 'data/test.npy'

    # Get train probs if not existed
    prob_file = 'checkpoint/train_osarisk_probs.npy'
    if not os.path.exists(prob_file):
        train_dataset = SleepSoundDataset(npy_file=train_npy)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        train_osarisk_probs = get_prob_list(train_loader, model, prob_file)
    else:
        train_osarisk_probs = np.load(prob_file)

    # Test dataset loader
    test_dataset = SleepSoundDataset(npy_file=test_npy)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Inference
    all_preds, all_labels, all_percentiles = inference(test_loader, model, train_osarisk_probs)

    print('\nPerformance on test set:')
    acc = (np.array(all_labels) == np.array(all_preds)).mean()
    print(f'Accuracy: {acc:.2%}')
    print(classification_report(all_labels, all_preds,
                                target_names=['No risk', 'OSA risk'], zero_division=0))

    np.set_printoptions(precision=4)
    print('\nCalculated percentiles:')
    print(np.array(sorted(all_percentiles)))
