import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from EDF import EDFReader

def extract_feats(mel_data: np.ndarray) -> np.ndarray:
    mean = mel_data.mean(axis=1)
    std = mel_data.std(axis=1)
    return np.concatenate((mean, std))

def load_and_preprocess_data(edf_file, label_file):
    # Load data
    mel_data = []
    edfObj = EDFReader(edf_file)
    for i in range(edfObj.meas_info['n_records']):
        data = edfObj.readBlock(i)
        data = data[4:]     # Take only Mel data
        mel_data.append(data)
    mel_data = np.concatenate(mel_data, axis=1)

    # Extract mean and std
    extracted_data = extract_feats(mel_data)

    # Load label
    with open(label_file) as f:
        label_data = json.load(f)
    label = label_data['Test_Result']['OSA_Risk']
    label = 1 if label == 'Y' else 0

    return extracted_data, label

def preprocess(root, csv_file, output_npy_file):
    csv_data = pd.read_csv(csv_file)
    data_paths = csv_data['filename'].tolist()
    data_paths = [os.path.join(root, x) for x in data_paths]

    # Read data
    all_extracted_data, all_label_data = [], []
    all_data_name = []
    for data_path in tqdm(data_paths, desc='Preprocessing data', ncols=120):
        data_name = os.path.basename(data_path)
        data_file = os.path.join(data_path, data_name + '-raw.edf')
        label_file = os.path.join(data_path, data_name + '.json')
        extracted_data, label_data = load_and_preprocess_data(data_file, label_file)
        all_extracted_data.append(extracted_data)
        all_label_data.append(label_data)
        all_data_name.append(data_name)

    # Save data
    save_dict = {'x': np.array(all_extracted_data),
                 'y': np.array(all_label_data),
                 'name': all_data_name}
    np.save(output_npy_file, save_dict)

if __name__ == '__main__':
    # Data root
    root = '/data/.data/20221215_modified'

    # Input CSV files
    train_csv = 'data_splits/sleep_train.csv'
    valid_csv = 'data_splits/sleep_valid.csv'
    test_csv = 'data_splits/sleep_test.csv'

    # Save path
    output_path = 'data'
    os.makedirs(output_path, exist_ok=False)

    # Preprocessing
    preprocess(root, train_csv, os.path.join(output_path, 'train.npy'))
    preprocess(root, valid_csv, os.path.join(output_path, 'valid.npy'))
    preprocess(root, test_csv, os.path.join(output_path, 'test.npy'))
