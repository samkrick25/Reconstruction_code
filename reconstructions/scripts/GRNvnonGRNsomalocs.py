from reconstructions.utils import load_data, preprocess_funcs
from reconstructions.utils.filedirs import freqspkl, allcoordswapped
import numpy as np
import pandas as pd
import pickle
from brainrender.actors import Points
from brainrender import Scene
import vedo

freqs = pickle.load(open(freqspkl, 'rb'))

latmerged = preprocess_funcs.merge_regions(freqs)
GRNposcells = [cell for cell in latmerged.index.tolist() if latmerged.loc[cell]['GRN'] > 3]
GRNnegcells = [cell for cell in latmerged.index.tolist() if latmerged.loc[cell]['GRN'] <= 3]

print(f'GRN positive: {GRNposcells}')
print(f'GRN negative: {GRNnegcells}')