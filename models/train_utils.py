import os, sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import configure_ffno3d
from models.functions.data_loader import GeologyTracesDataset

def get_model(options):
    if options['model_type'] == "ffno3d":
        return configure_ffno3d.get_model(options)
    else:
        raise TypeError("No such model type: " + options['model_type'])
    
def get_dataset(options, train_or_val: str):
    if train_or_val != "train" and train_or_val != "val":
        raise TypeError("Dataset type must be train or val")

    path_data = './inputs/'
    
    if train_or_val == "train":
        dir = options['dir_data_train']
        num_data = options['Ntrain']
    else:
        dir = options['dir_data_val']
        num_data = options["Nval"]

    return GeologyTracesDataset(path_data, dir, options['T_out'], options['S_in'],
                                options['S_in_z'], options['S_out'], 'normal', num_data)

def get_dataloader(dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        pin_memory = True,
        shuffle = False,
        sampler = DistributedSampler(dataset)
    )

def get_name_config(options):
    dv = options['dv']
    S_in = options['S_in']
    T_out = options['T_out']
    padding = options['padding']
    learning_rate = options['learning_rate']
    loss_weights = options['loss_weights']
    Ntrain = options['Ntrain']
    batch_size = options['batch_size']
    model_type = options['model_type']

    name_config = f"{model_type}-dv{dv}-{options['nlayers']}layers-S{S_in}-T{T_out}-padding{padding}-learningrate{str(learning_rate).replace('.','p')}-" \
                f"L1loss{str(loss_weights[0]).replace('.','p')}-L2loss{str(loss_weights[1]).replace('.','p')}-"
    name_config += f"Ntrain{Ntrain}-batchsize{batch_size}"
    name_config += options['additional_name']

    return name_config