import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.nn.functional import softmax

from data_preprocess import extract_feats
from EDF import EDFReader
from model import NeuralNetwork
from utils import load_model


def get_percentile(probs, train_osarisk_probs):
    percentiles = []
    for prob in probs:
        cnt_larger = (train_osarisk_probs >= prob).sum()
        percentile_val = (cnt_larger + 1) / len(train_osarisk_probs)
        percentiles.append(percentile_val)
    return percentiles

def get_prob_percentile(model, batch_data, train_osarisk_probs):
    # Probabilities
    logits = model(batch_data)
    probs = softmax(logits, dim=1)
    probs = probs[:, 1].tolist()    # Probability of having OSA risk

    # Percentiles
    percentiles = get_percentile(probs, train_osarisk_probs)

    return probs, percentiles

def read_mel_data(input_edf_file):
    try:
        # Read mel data
        mel_data = []
        edfObj = EDFReader(input_edf_file)
        for i in range(edfObj.meas_info['n_records']):
            data = edfObj.readBlock(i)
            data = data[4:]     # Take only Mel data
            mel_data.append(data)
        mel_data = np.concatenate(mel_data, axis=1)
        return mel_data

    except:
        # Return error
        print(f'Error reading EDF file: {input_edf_file}')
        return False

def preprocess(input_edf_file):
    # Read mel data
    mel_data = read_mel_data(input_edf_file)
    if type(mel_data) == bool and (mel_data == False):
        return False

    # Extract mean and std
    extracted_data = extract_feats(mel_data)

    return extracted_data

def inference(extracted_data, model, train_osarisk_probs):
    with torch.no_grad():
        # Data preparation
        X = extracted_data[np.newaxis, :]   # Batch size 1
        X = torch.from_numpy(X).float()
        X = X.to(device)

        # Inference
        prob, percentile = get_prob_percentile(model, X, train_osarisk_probs)

    return prob[0], percentile[0]

def nia_api(input_edf_file):
    # Data processing
    extracted_data = preprocess(input_edf_file)
    if type(extracted_data) == bool and (extracted_data == False):
        # Error happened
        error = True
        return None, None, None, error

    # Load model
    model = NeuralNetwork().to(device)
    model_pt_file = 'checkpoint/best_model.pt'
    load_model(model, model_file=model_pt_file)
    model.eval()

    # Get probability
    prob_file = 'checkpoint/train_osarisk_probs.npy'
    train_osarisk_probs = np.load(prob_file)

    # Get probabilities and percentiles
    prob, percentile = inference(extracted_data, model, train_osarisk_probs)
    result = True if prob >= 0.5 else False
    error = False

    return result, prob, percentile, error

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    test_file = '/data/.data/20221215_modified/C2022-VD-04-0334/C2022-VD-04-0334-raw.edf'

    # NIA API
    start = datetime.now()
    result, prob, percentile, error = nia_api(test_file)
    print('Elapsed time:', datetime.now() - start)
    print('\nResult:', result)
    print('Probability:', prob)
    print('Percentile:', percentile)
    print('Error:', error)
