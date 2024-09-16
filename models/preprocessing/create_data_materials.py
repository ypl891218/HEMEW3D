import numpy as np
import argparse
import h5py

parser = argparse.ArgumentParser(prefix_chars='@')
parser.add_argument('@Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('@Nval', type=int, default=3000, help="Number of test samples")
parser.add_argument('@Irun0', type=int, default=100000, help="Offset index")
parser.add_argument('@S_in', type=int, default=32, help="Size of the geology for the network")
parser.add_argument('@S_in_z', type=int, default=32, help="Depth of the geology for the network")
parser.add_argument('@Nt', type=int, default=320, help='Number of time steps in the output (only used for the folder name here)')
parser.add_argument('@fmax', type=int, default=5, help='Maximum frequency for filtering (only used for the folder name here)')
options = parser.parse_args().__dict__

S_in = options['S_in'] # size of the geology (x and y) for the network input
S_in_z = options['S_in_z']
Nt = options['Nt']
Ntrain = options['Ntrain']
Nval = options['Nval']
Irun0 = options['Irun0']
fmax = options['fmax']

path_save = '../inputs/' # where to save the machine learning inputs
path_a = '../../data/' # path to original raw data
folder_name = f'inputs3D_S{S_in}_Z{S_in_z}_T{Nt}_fmax{fmax}'

## Modified by lyp

npy_files = ['material100000-101999.npy', 'material102000-103999.npy', 'material104000-105999.npy', 'material106000-107999.npy', 'material108000-109999.npy',
            'material110000-111999.npy', 'material112000-113999.npy', 'material114000-115999.npy', 'material116000-117999.npy', 'material118000-119999.npy',
             'material120000-121999.npy', 'material122000-123999.npy', 'material124000-125999.npy', 'material126000-127999.npy', 'material128000-129999.npy']
N = 2000
file_count = 0

for file_count, npy_file in enumerate(npy_files):
    # Load materials
    data_a = np.load(path_a + npy_file)
    data_a = data_a[:Ntrain+Nval, :32, :32, :32]
    data_a = data_a.astype(np.float32)

    base_idx = file_count * N

    # Save data to individual .h5 files
    for i in range(base_idx, base_idx+N):
        if i < Ntrain:
            with h5py.File(f'{path_save}{folder_name}_train/sample{Irun0+i}.h5', 'w') as f:
                f.create_dataset('a', data=data_a[i-base_idx])
        else:
            with h5py.File(f'{path_save}{folder_name}_val/sample{Irun0+i}.h5', 'w') as f:
                f.create_dataset('a', data=data_a[i-base_idx])
    print("Finish {} to {}".format(base_idx, base_idx+N-1), flush=True)
