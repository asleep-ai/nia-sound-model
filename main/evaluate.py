import os

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from dataset import SleepSoundDataset
from model import NeuralNetwork
from utils import set_seed, load_model


def inference(test_loader, model):
    all_preds, all_labels, all_names, all_confidences = [], [], [], []
    with torch.no_grad():
        for X, y, names in test_loader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            probs = softmax(logits, dim=1)

            all_preds += preds.tolist()
            all_labels += y.tolist()
            all_names += list(names)
            all_confidences += probs[:, 1].tolist()     # Prob of OSA risk
    return all_preds, all_labels, all_names, all_confidences


if __name__ == '__main__':
    set_seed(seed=0)

    batch_size = 8
    train_npy = 'data/train.npy'
    test_npy = 'data/test.npy'
    exp_path = 'checkpoint'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Load model
    model = NeuralNetwork().to(device)
    model_pt_file = os.path.join(exp_path, 'best_model.pt')
    load_model(model, model_file=model_pt_file)
    model.eval()

    # Test dataset loader
    test_dataset = SleepSoundDataset(npy_file=test_npy, get_name=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Inference
    all_preds, all_labels, all_names, all_confidences = inference(test_loader, model)

    print('\nPerformance on test set:')
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    acc = (np.array(all_labels) == np.array(all_preds)).mean()
    print(f'Accuracy: {acc:.2%}')
    print(classification_report(all_labels, all_preds,
                                target_names=['No risk', 'OSA risk'], zero_division=0))

    print('\nWriting results to a CSV file...')
    data = {
        'Data ID': all_names,
        'Ground Truth': all_labels,
        'Prediction': all_preds,
        'Confidence': all_confidences
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(exp_path, 'results.csv'), index=False)
    print('Done!')
