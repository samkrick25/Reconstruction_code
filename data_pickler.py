from reconstructions.utils import load_data as ld
import pickle
from reconstructions.utils.filedirs import pterm, allcoordswapped, allen_ccf_10um, allen_parcellation, allen_2017_to_2020, parcellated_neurons
import nibabel as nib
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
# =============================================================================
# 
# _, somas, _, _, _ = ld.load_neurons(allcoordswapped)
# # = load_data.get_frequencies(cells, somas)
# 
# # savefile = r'reconstructions\data\freqs.pkl'
# # pickle.dump(freqs, open(savefile, 'wb'))
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

neurons = ld.load_neurons(allcoordswapped)

frequencies = {}
for neuron, info in tqdm(neurons.items(), desc='finding endpoint frequencies'):
    print(neuron)
    freqs = ld.neuronprop({neuron: info}, mode='frequency', ontlevel='structure', cellname=neuron)
    frequencies[neuron] = freqs

fdf = pd.DataFrame(frequencies)
fdf = fdf.replace(np.nan, 0)
savefile = r'reconstructions\data\freqs_hopefullygood.pkl'
pickle.dump(fdf, open(savefile, 'wb'))

# =============================================================================
# frequencies = {}
# for file in tqdm(os.listdir(parcellated_neurons), desc='Finding frequencies'):
#     filename = os.path.join(parcellated_neurons, file)
#     with open(filename, 'r') as f:
#         neurondict = json.load(f)
#         freqseries = ld.get_frequencies_from_dict(neurondict, ontlevel='structure')
#     frequencies[freqseries.name] = freqseries
# fdf = pd.DataFrame(frequencies)
# frequencies_nonan = fdf.replace(np.nan, 0)
# savefile = r'reconstructions\data\frequencies2.pkl'
# pickle.dump(frequencies_nonan, open(savefile, 'wb'))
# =============================================================================
# =============================================================================
# ptermsave = r'reconstructions/data/pterm.pkl'
# ptermdf = pd.read_csv(pterm, index_col='label')
# pickle.dump(ptermdf, open(ptermsave, 'wb'))
# =============================================================================
#neurons = ld.load_neurons(allcoordswapped)

lengthdict = {}
for neuron, info in tqdm(neurons.items(), desc='finding lengths'):
    lengths = ld.neuronprop({neuron:info}, mode='length', ontlevel='structure', cellname=neuron)
    ser = pd.Series(lengths, name=neuron)
    lengthdict[neuron] = ser
    
lengthdf = pd.DataFrame(lengthdict)
lengthdf = lengthdf.replace(np.nan, 0)
savefile = r'reconstructions\data\lengths_hopefullygood.pkl'
pickle.dump(lengthdf, open(savefile, 'wb'))

