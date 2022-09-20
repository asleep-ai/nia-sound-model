import os
import sys
sys.path.append('../one_cycle/')

from EDF import EDFReader


def mel_read(edf_file):
    edfObj = EDFReader(edf_file)
    for i in range(edfObj.meas_info['n_records']):
        try:
            data = edfObj.readBlock(i)
        except:
            return False
    return True

if __name__ == '__main__':
    root = '/HDD/nia/data_backup'
    edf_files = [os.path.join(root, x) for x in os.listdir(root) if x.endswith('.edf')]

    for edf_file in edf_files:
        readable = mel_read(edf_file)
        if not readable:
            print(edf_file)
