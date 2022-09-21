import json
import os

def list_ids(label_path):
    label_file_list = [x for x in os.listdir(label_path) if x.endswith('.json')]
    id_list = [x[:-5] for x in label_file_list]
    return id_list

if __name__ == '__main__':
    data_dir = '/HDD/nia/data'

    # List of IDs for each part
    train_ids = list_ids(os.path.join(data_dir, 'train'))
    test_ids = list_ids(os.path.join(data_dir, 'test'))
    corrupted_ids = list_ids(data_dir)

    # Prepare JSON formatted str
    idDict = {
        'train': train_ids,
        'test': test_ids,
        'corrupted': corrupted_ids
    }
    jsonString = json.dumps(idDict, indent=4)

    with open('data.json', 'w') as jsonFile:
        jsonFile.write(jsonString)
