from reconstructions.utils import load_data as ld
import pickle
from reconstructions.utils.filedirs import allcoordswapped, allen_ccf_10um, allen_parcellation, allen_2017_to_2020
import nibabel as nib
import pandas as pd
import json
import os
import numpy as np

# =============================================================================
# cells, somas, _, _, _ = load_data.load_neurons(allcoordswapped)
# freqs = load_data.get_frequencies(cells, somas)
# 
# savefile = r'reconstructions\data\freqs.pkl'
# pickle.dump(freqs, open(savefile, 'wb'))
# 
# savefilesomas = r'reconstructions\data\somas.pkl'
# pickle.dump(somas, open(savefilesomas, 'wb'))
# =============================================================================

# =============================================================================
# save1 = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\parcellation.pkl'
# save2 = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\parcellation_map.pkl'
# allen_parcellation = pd.read_csv(allen_parcellation, index_col='parcellation_index')
# parcellation_map = pd.read_csv(allen_2017_to_2020, index_col='parcellation_label')
# pickle.dump(allen_parcellation, open(save1, 'wb'))
# pickle.dump(parcellation_map, open(save2, 'wb'))
# 
# =============================================================================

neuronsdict = ld.load_neurons(allcoordswapped)
ld.get_node_parcellations(neuronsdict)
savedir = r'reconstructions\data\IRNPARN_cells\json_parcellated'
for cell, info in neuronsdict.items():
    savefile=os.path.join(savedir, cell + '.json')
    neuron = {cell: info}
    with open(savefile, 'w') as f:
        json.dump(neuron, f)
#savedict = r'reconstructions\data\neurondict.pkl'
#pickle.dump(neurondict, open(savedict, 'wb'))