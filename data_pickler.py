from reconstructions.utils import load_data as ld
import pickle
from reconstructions.utils.filedirs import allcoordswapped, allen_ccf_10um, allen_parcellation, allen_2017_to_2020, parcellated_neurons
import nibabel as nib
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

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

# =============================================================================
# neuronsdict = ld.load_neurons(allcoordswapped)
# ld.get_node_parcellations(neuronsdict)
# savedir = r'reconstructions\data\IRNPARN_cells\json_parcellated'
# for cell, info in neuronsdict.items():
#     savefile=os.path.join(savedir, cell + '.json')
#     neuron = {cell: info}
#     with open(savefile, 'w') as f:
#         json.dump(neuron, f)
# =============================================================================
#savedict = r'reconstructions\data\neurondict.pkl'
#pickle.dump(neurondict, open(savedict, 'wb'))

frequencies = pd.DataFrame()
for file in tqdm(os.listdir(parcellated_neurons), desc='Finding frequencies'):
    filename = os.path.join(parcellated_neurons, file)
    with open(filename, 'r') as f:
        neurondict = json.load(f)
        freqseries = ld.get_frequencies_from_dict(neurondict, ontlevel='structure')
    frequencies = pd.concat([frequencies, freqseries], join='outer', axis=1)
frequencies_nonan = frequencies.replace(np.nan, 0)
savefile = r'reconstructions\data\frequencies.pkl'
pickle.dump(frequencies_nonan, open(savefile, 'wb'))