import os, sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from models.functions.ffno_model import FFNO_3D

def get_model(options):
    list_D1 = np.array(options['list_D1']).astype(int)
    list_D2 = np.array(options['list_D2']).astype(int)
    list_D3 = np.array(options['list_D3']).astype(int)
    list_M1 = np.array(options['list_M1']).astype(int)
    list_M2 = np.array(options['list_M2']).astype(int)
    list_M3 = np.array(options['list_M3']).astype(int)
    nlayers = options['nlayers']
    dv = options['dv']

    assert nlayers == list_D1.shape[0]
    
    model = FFNO_3D(list_D1, list_D2, list_D3,
                    list_M1, list_M2, list_M3, dv, 
                    input_dim=4, # to define the uplift network (last dimension after grid concatenation)
                    output_dim=1, # to define the projection network (last dimension after projection)
                    n_layers=nlayers,
                    padding = 0
    )  # load model
    return model
