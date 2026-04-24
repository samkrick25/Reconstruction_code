# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:40:35 2026

@author: economolab
"""
from reconstructions.utils.filedirs import parcellation_mappkl, structure_ont_info, frequenciespkl
from reconstructions.utils import preprocess_funcs
import numpy as np
import pandas as pd
#import scipy
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import pickle
import seaborn as sns
#import plotly.express as px
#TODO: rewrite using plotly as it allows me to zoom around and investigate clusters at a closer resolution, although i don't have time to do this rn
#have to find a way to get my row colors to work

# %%

# %%
#os.chdir(r'C:\Users\economolab\Documents\GitHub\Reconstruction_code')
structure_to_ont = pickle.load(open(structure_ont_info, 'rb'))
parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))
frequencies_notprocessed = pickle.load(open(frequenciespkl, 'rb'))
frequencies = preprocess_funcs.preprocess(frequencies_notprocessed)
# %%

divisiondf = parcellation_map.loc[(parcellation_map['parcellation_term_set_name']=='division')]
uniquedivisions = np.unique(divisiondf['parcellation_term_acronym'].values)

hexvals = {}
for division in uniquedivisions:
    hexvals[division] = divisiondf.loc[divisiondf['parcellation_term_acronym']==division, 'color_hex_triplet'].values[0]

ontology_order = {'AQ':0, 'Isocortex':1, 'CTXsp':2, 'HPF':3, 'OLF':4, 'CB':5, 'PAL':6, 'STR':7, 'TH':8, 'HY':9, 'MB':10, 'P':11, 'MY':12,
                  'V3':13,'V4':14, 'VL':15, 'brain-unassigned':16, 'c':17, 'cbf':18, 'cm':19, 'eps':20, 'fiber tracts-unassigned':21,'lfbs':22,
                  'mfbs':23, 'scwm':24, 'unassigned':25}
laterality_order = {'Ipsilateral':0, 'Contralateral':1}

index_df = pd.DataFrame(index=frequencies.index, columns=['regionDivision', 'laterality', 'meanProjection'])
for region in index_df.index:
    regstrparts = region.split()[1:]
    lat = region.split()[0]
    index_df.loc[region, 'laterality'] = lat
    regstr = ' '.join(regstrparts)
    ontinfo = structure_to_ont[regstr]
    division = ontinfo['acronymInfo']['division']
    index_df.loc[region, 'regionDivision'] = division
    
index_df['meanProjection'] = frequencies.mean(axis=1).values
index_df['lateralityOrder'] = index_df['laterality'].map(laterality_order)
index_df['ontologyOrder'] = index_df['regionDivision'].map(ontology_order)

sorted_regions = index_df.sort_values(by=['lateralityOrder', 'ontologyOrder', 'meanProjection'], ascending=[True, True, False])
freq_sorted = frequencies.reindex(sorted_regions.index)
row_colors = [hexvals[division] for division in sorted_regions['regionDivision']]

#setting colormap, 0 values set to black
vals = freq_sorted.values
nonzero = vals[vals != 0]
vmin = nonzero.min() if nonzero.size>0 else 0
vmax = nonzero.max() if nonzero.size>0 else 1
base_cmap = plt.cm.nipy_spectral
bounds = [0, 0.00001] + list(np.linspace(vmin, vmax, 256))
zero_color = (np.float64(0),np.float64(0),np.float64(0),np.float64(1))
colors = [zero_color] + [base_cmap(i) for i in range(base_cmap.N)]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)

g = sns.clustermap(freq_sorted,
        method='ward',
        metric='euclidean',
        cmap=cmap,
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
# =============================================================================
# #this is all to set the colormap for my rows during clustering, I want my rows sorted and colored by ontology, but had to manually
# #annotate to get the resolution I desired
# annot = pd.read_csv(r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\region_list_for_annotation.csv')
# #frequency_tocluster['Mean'] = np.mean(frequency_tocluster, axis=1)
# ont_lut = {
#     'CTX': "#12ff0a",  
#     'HY': "#c70000",   
#     'TH': "#ff7676",   
#     'MB': "#ff01ea",   
#     'CB': "#fffb02",   
#     'P-sen': "#ffa048",
#     'P-mot': "#fc802e",
#     'P-sat': "#ffba7a",
#     'P': "#fd6500",    
#     'MY-mot': "#f574ce",
#     'MY-sen': "#ff98e0",
#     'MY-sat': "#faafe3",
#     'MY': "#f84ac4",
#     'Other':  "#ada798fd",
#     'CNU': "#00fff2fc"  
# }
# 
# index_df = pd.DataFrame({'FullRegion': frequencies.index})
# 
# index_df = index_df.merge(annot, left_on='FullRegion', right_on='Region', how='left')
# index_df[['Laterality', 'Region']] = index_df['Region'].str.extract(r'^(Ipsilateral|Contralateral) (.+)$')
# index_df['MeanProjection'] = frequencies.mean(axis=1).values
# 
# laterality_order = {'Ipsilateral': 0, 'Contralateral': 1}
# index_df['LateralityOrder'] = index_df['Laterality'].map(laterality_order)
# 
# ontology_order = {'CTX':0, 'CNU':1, 'HY':2, 'TH':3, 'MB':4, 'CB':5, 'P':6, 'P-sen':7, 'P-mot':8,
#                  'P-sat':9, 'MY':10, 'MY-mot':11, 'MY-sen':12, 'MY-sat':13, 'Other':14}  # adjust for desired ontology levels
# index_df['OntologyOrder'] = index_df['Ontology'].map(ontology_order)
# sorted_index = index_df.sort_values(by=['LateralityOrder', 'OntologyOrder', 'MeanProjection'], ascending=[True, True, False])
# #sorted_index['color'] = ont_lut[]
# freq_sorted = frequencies.reindex(sorted_index['FullRegion'])
# #print(freq_sorted)
# row_colors = [ont_lut[cat] for cat in sorted_index['Ontology']]
# vals = freq_sorted.values
# nonzero = vals[vals != 0]
# vmin = nonzero.min() if nonzero.size>0 else 0
# vmax = nonzero.max() if nonzero.size>0 else 1
# base_cmap = plt.cm.gist_earth
# bounds = [0, 0.00001] + list(np.linspace(vmin, vmax, 256))
# zero_color = (np.float64(0),np.float64(0),np.float64(0),np.float64(1))
# colors = [zero_color] + [base_cmap(i) for i in range(base_cmap.N)]
# cmap = mcolors.ListedColormap(colors)
# norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)
# print('Clustering now:')
# g = sns.clustermap(freq_sorted,
#         method='ward',
#         metric='euclidean',
#         cmap = cmap,
#         norm=norm,
#         row_cluster=False,
#         col_cluster=True,
#         row_colors=row_colors,
#         dendrogram_ratio=(.1, .15),
#         cbar_pos=(0, .15, .03, .7),
#         cbar_kws={'label': 'ln(percent of terminals in region + 1)'},
#         yticklabels=False,
#         xticklabels=True,
#         figsize=(18, 18),         # wider figure for right axis
# )
# cbar_ax = g.cax
# cbar_ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%d'))
# cbar_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
# g.figure.canvas.draw()
# #g.fig.show()
# savepath = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\recons_clustermap_parcellated.png'
# g.savefig(savepath, dpi=300, bbox_inches='tight')
# =============================================================================
