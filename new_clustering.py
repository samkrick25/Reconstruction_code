# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:40:35 2026

@author: economolab
"""

from reconstructions.utils import load_data
from reconstructions.utils.filedirs import neurondictpkl
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import time
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

#os.chdir(r'C:\Users\economolab\Documents\GitHub\Reconstruction_code')

pklstart = time.perf_counter()
neurondict = pickle.load(open(neurondictpkl, 'rb'))
pklend = time.perf_counter()
print(f'Time to load neurondict: {pklend-pklstart}')
frequencies = load_data.get_frequencies_from_dict(neurondict, ontlevel='structure')

#this is all to set the colormap for my rows during clustering, I want my rows sorted and colored by ontology, but had to manually
#annotate to get the resolution I desired
annot = pd.read_csv(r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\region_list_for_annotation.csv')
#frequency_tocluster['Mean'] = np.mean(frequency_tocluster, axis=1)
ont_lut = {
    'CTX': "#12ff0a",  
    'HY': "#c70000",   
    'TH': "#ff7676",   
    'MB': "#ff01ea",   
    'CB': "#fffb02",   
    'P-sen': "#ffa048",
    'P-mot': "#fc802e",
    'P-sat': "#ffba7a",
    'P': "#fd6500",    
    'MY-mot': "#f574ce",
    'MY-sen': "#ff98e0",
    'MY-sat': "#faafe3",
    'MY': "#f84ac4",
    'Other':  "#ada798fd",
    'CNU': "#00fff2fc"  
}

index_df = pd.DataFrame({'FullRegion': frequencies.index})

index_df = index_df.merge(annot, left_on='FullRegion', right_on='Region', how='left')
index_df[['Laterality', 'Region']] = index_df['Region'].str.extract(r'^(Ipsilateral|Contralateral) (.+)$')
index_df['MeanProjection'] = frequencies.mean(axis=1).values

laterality_order = {'Ipsilateral': 0, 'Contralateral': 1}
index_df['LateralityOrder'] = index_df['Laterality'].map(laterality_order)

ontology_order = {'CTX':0, 'CNU':1, 'HY':2, 'TH':3, 'MB':4, 'CB':5, 'P':6, 'P-sen':7, 'P-mot':8,
                 'P-sat':9, 'MY':10, 'MY-mot':11, 'MY-sen':12, 'MY-sat':13, 'Other':14}  # adjust for your ontology levels
index_df['OntologyOrder'] = index_df['Ontology'].map(ontology_order)
sorted_index = index_df.sort_values(by=['LateralityOrder', 'OntologyOrder', 'MeanProjection'], ascending=[True, True, False])
#sorted_index['color'] = ont_lut[]
freq_sorted = frequencies.reindex(sorted_index['FullRegion'])
#print(freq_sorted)
row_colors = [ont_lut[cat] for cat in sorted_index['Ontology']]
vals = freq_sorted.values
nonzero = vals[vals != 0]
vmin = nonzero.min() if nonzero.size>0 else 0
vmax = nonzero.max() if nonzero.size>0 else 1
base_cmap = plt.cm.gist_earth
bounds = [0, 0.00001] + list(np.linspace(vmin, vmax, 256))
zero_color = (np.float64(0),np.float64(0),np.float64(0),np.float64(1))
colors = [zero_color] + [base_cmap(i) for i in range(base_cmap.N)]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)
print('Clustering now:')
g = sns.clustermap(freq_sorted,
        method='ward',
        metric='euclidean',
        cmap = cmap,
        norm=norm,
        row_cluster=False,
        col_cluster=True,
        row_colors=row_colors,
        dendrogram_ratio=(.1, .15),
        cbar_pos=(0, .15, .03, .7),
        cbar_kws={'label': 'ln(percent of terminals in region + 1)'},
        yticklabels=False,
        xticklabels=True,
        figsize=(18, 18),         # wider figure for right axis
)
cbar_ax = g.cax
cbar_ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%d'))
cbar_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
g.figure.canvas.draw()
#g.fig.show()
savepath = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\recons_clustermap_parcellated.png'
g.savefig(savepath, dpi=300, bbox_inches='tight')
