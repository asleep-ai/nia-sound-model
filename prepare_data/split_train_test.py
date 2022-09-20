import json
import os
import subprocess

import numpy as np

# Label files that have corrupted corresponding EDF data
WRONG_DATA = [
    'c020wwej_2022-08-10_225855-3',
    'c386rhze_2022-08-18_020759-2',
    'c397wmmc_2022-08-16_021357-4',
    'c010nrkz_2022-07-29_044907-2',
]

def read_labels(label_file):
    with open(label_file) as f:
        label_data = json.load(f)
    label = label_data['Test_Result']['OSA_Risk']
    label = 1 if label == 'Y' else 0
    return label

def get_all_labels(label_file_list):
    labels = []
    positive_files = []
    negative_files = []
    for label_file in label_file_list:
        if os.path.basename(label_file)[:-5] in WRONG_DATA:
            continue
        label = read_labels(label_file)
        labels.append(label)
        if label == 0:
            negative_files.append(label_file)
        elif label == 1:
            positive_files.append(label_file)
        else:
            raise ValueError(f'Unsupported value of label: label = {label} for JSON file {label_file}')

    labels = np.array(labels)
    return labels, positive_files, negative_files

def split(file_list, test_portion):
    n_test = int(len(file_list) * test_portion)
    n_train = len(file_list) - n_test
    assert n_train > n_test
    test_list = np.random.choice(file_list, n_test, replace=False)
    train_list = [x for x in file_list if x not in test_list]
    return train_list, test_list.tolist()

def move(label_file_list, target_folder):
    for label_file in label_file_list:
        data_file = label_file[:-5] + '-raw.edf'
        assert os.path.exists(data_file), f'{data_file} does not exist.'
        assert os.path.exists(label_file), f'{label_file} does not exist.'

        # Data file
        try:
            subprocess.run(['mv', data_file, target_folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        except:
            print(f'There is a problem moving {data_file}.')

        # Label file
        try:
            subprocess.run(['mv', label_file, target_folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        except:
            print(f'There is a problem moving {data_file}.')


if __name__ == '__main__':
    root_dir = '/HDD/nia/data'
    label_file_list = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if x.endswith('.json')]
    label_data, positive_files, negative_files = get_all_labels(label_file_list)

    # Statistics
    n_positive = (label_data == 1).sum()
    n_negative = (label_data == 0).sum()
    print(f'n_positive = {n_positive}, n_negative = {n_negative}')

    # Split
    TEST_PORTION = 0.2
    positive_train_list, positive_test_list = split(positive_files, TEST_PORTION)
    negative_train_list, negative_test_list = split(negative_files, TEST_PORTION)
    train_list = positive_train_list + negative_train_list
    test_list = positive_test_list + negative_test_list

    print(f'len(positive_train_list) = {len(positive_train_list)}')
    print(f'len(negative_train_list) = {len(negative_train_list)}')
    print(f'len(train_list) = {len(train_list)}')
    print(f'len(positive_test_list) = {len(positive_test_list)}')
    print(f'len(negative_test_list) = {len(negative_test_list)}')
    print(f'len(test_list) = {len(test_list)}')

    # Move training files
    target_dir_train = os.path.join(root_dir, 'train')
    os.makedirs(target_dir_train, exist_ok=True)
    move(train_list, target_dir_train)

    # Move test files
    target_dir_test = os.path.join(root_dir, 'test')
    os.makedirs(target_dir_test, exist_ok=True)
    move(test_list, target_dir_test)
