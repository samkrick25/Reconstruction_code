# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:00:36 2026
premotor heatmap
@author: samkr
"""

from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import preprocess_funcs as pp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import matplotlib.colors as mcolors

thresh=10

cols = ['XII', 'V', 'VII']
pattern_order = ['100', '010', '001', '110', '101', '011', '111']

def get_pattern(row):
    """Create binary pattern (1=above threshold, 0=zero)"""
    return ''.join(['1' if val >= thresh else '0' for val in row])

def get_max_in_pattern(row):
    """Get max value among columns marked as 1 in the pattern"""
    pattern = row['_pattern']
    max_val = 0
    for i, bit in enumerate(pattern):
        if bit == '1':
            max_val = max(max_val, row[cols[i]])
    return max_val

frequencies = pickle.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(frequencies)

premotor = merged.loc[(merged['XII']>thresh) | (merged['VII']>thresh) | (merged['V']>thresh)]

premotorMNs = premotor[['XII', 'V', 'VII']]

premotorMNs['_pattern'] = premotorMNs.apply(get_pattern, axis=1)
premotorMNs['_max_val'] = premotorMNs.apply(get_max_in_pattern, axis=1)

premotorMNs['_pattern'] = pd.Categorical(premotorMNs['_pattern'], categories=pattern_order, ordered=True)
premotorSorted = premotorMNs.sort_values(['_pattern', '_max_val'], ascending = [True, False])

premotorSorted = premotorSorted.drop(columns=['_pattern', '_max_val'])

premotorLog = pp.preprocess(premotorSorted, pct=False)

# =============================================================================
# vals = premotorLog.values
# nonzero = vals[vals != 0]
# vmin = nonzero.min() if nonzero.size>0 else 0
# vmax = nonzero.max() if nonzero.size>0 else 1
# base_cmap = plt.cm.inferno
# bounds = [0, 0.00001] + list(np.linspace(vmin, vmax, 256))
# zero_color = (np.float64(0),np.float64(0),np.float64(0),np.float64(1))
# colors = [zero_color] + [base_cmap(i) for i in range(base_cmap.N)]
# cmap = mcolors.ListedColormap(colors)
# norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)
# =============================================================================
#, cmap='jet'
fig, ax = plt.subplots(figsize=(20, 3),dpi=300)
sns.heatmap(premotorLog.T, ax=ax, xticklabels=premotorSorted.index, cbar_kws={'location':'left', 'label':'ln(# of endpoints + 1)'})
