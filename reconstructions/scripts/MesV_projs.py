import pandas as pd
from brainrender import Scene
from brainrender.actors import Neuron
from reconstructions.utils.filedirs import freqspkl, somaspkl
import pickle
from reconstructions.utils import load_data, preprocess_funcs

#just want to list cells that have projections in MEV or to surrounding areas (PAG, SCm, MRN, MB)

freqs = pickle.load(open(freqspkl, 'rb'))

merged = preprocess_funcs.merge_regions(freqs)

rois = ['MEV', 'PAG', 'MB', 'MRN', 'SCiw', 'SCdg', 'SCig']
mergedrois = merged[rois]

roiprojcells = [cell for cell, row in mergedrois.iterrows() if (row != 0).any()]
print(roiprojcells)