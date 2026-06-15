# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:08:39 2026
make P(MRN|GRN) v P(MRN|~GRN)
@author: samkr
"""
from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import preprocess_funcs as pp
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib_venn import venn2

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'arial'

THRESH = 2

frequencies = pickle.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(frequencies)

NCELLS = len(merged.index)

masked = merged.astype(bool)
pGRN = masked.sum(axis=0)['GRN'] / NCELLS
pMRN = masked.sum(axis=0)['MRN'] / NCELLS

MRNdf = merged.loc[merged['MRN'] != 0]
nonMRNdf = merged.loc[merged['MRN'] == 0]
GRNdf = merged.loc[merged['GRN'] != 0]

MRNmask = MRNdf.astype(bool)
pGRNMRN = MRNmask.sum(axis=0)['GRN']/len(MRNmask.index)

pMRNGRN = (pGRNMRN*pMRN)/pGRN

nonGRNdf = merged.loc[merged['GRN'] == 0]
nonGRNmask = nonGRNdf.astype(bool)
pMRNnonGRN = nonGRNmask.sum(axis=0)['MRN']/len(nonGRNmask.index)

GRNMRN = MRNmask.sum(axis=0)['GRN']
GRNmrn = len(GRNdf.index) - GRNMRN

grnMRN = len(MRNmask['GRN']) - GRNMRN

labels = ['GRN', 'GRN&MRN', 'MRN&~GRN']
colors = ['blue', 'darkorange']
subsets = {'10':GRNmrn, '01':grnMRN, '11':GRNMRN}
#fig, ax = plt.subplots(dpi=300)
venn = venn2(subsets=subsets, set_labels=labels, set_colors=colors)
plt.savefig(r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\GRNMRNvenn.svg')


# =============================================================================
# toPlot = [pGRNMRN, pMRNnonGRN]
# labels = ['P(GRN|MRN)' , 'P(MRN|~GRN)']
# colors = ['green', 'blue']
# 
# fig, ax = plt.subplots(1, 1, dpi=300)
# ax.bar(labels, toPlot, color=colors)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_ylabel('Probability')
# ax.set_yticks(np.arange(0.0, 1.25, .25))
# fig.savefig(r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\GRNMRN.svg')
# =============================================================================
