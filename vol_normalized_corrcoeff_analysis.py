# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:37:33 2026
innervation of regions looking at axon length/mm^3
@author: samkr
"""

from reconstructions.utils.filedirs import lengthspkl, ccf_vols3, frequenciespkl, structure_ont_info
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from reconstructions.utils import preprocess_funcs as pp
import csv
import pandas as pd
import scipy.cluster.hierarchy as sch

def to_mm(series):
    return series.to_list()/np.float64(1000)

def divide_by_vol(series):
    regname = series.name
    regvol = structure_vols[regname]
    return series.to_list()/regvol

def normalize(df, mm=False, div=True):
    if mm == True:
        df = df.apply(to_mm, axis=0)
    return df.apply(divide_by_vol, axis=0)
# %%
#
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.filedirs import lengthspkl, ccf_vols3, frequenciespkl, structure_ont_info
structure_to_ont = pickle.load(open(structure_ont_info, 'rb'))
freqs = pickle.load(open(r'reconstructions\data\freqs_hopefullygood.pkl', 'rb')).T
lengths = pickle.load(open(r'reconstructions\data\lengths_hopefullygood.pkl', 'rb')).T
lmerged = pp.merge_regions(lengths)
fmerged = pp.merge_regions(freqs)
lnozero = lmerged.loc[:, (lmerged != 0).any(axis=0)]
fnozero = fmerged.loc[:, (fmerged != 0).any(axis=0)]

#pp.write_targeted_regions_to_excel(freqs, r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\targeted_reg_lat.xlsx')
# %%


structure_vols = {}
with open(ccf_vols3, 'r') as svols:
    reader = csv.reader(svols)
    for row in reader:
        structure_vols[row[0]] = np.float64(row[1])

# =============================================================================
# fregtodrop = ['SpC', 'VL-unassigned', 'mfbc', 'ECT', 'SSs', 'LSc', 'VISC', 'EPd', 'IA', 'TT',
#        'ACB', 'epsc', 'DP', 'cc', 'AIp', 'LSv', 'BMA', 'BST', 'MPO', 'ADP', 'LSv', 'BMA', 'BST', 'MPO', 'ADP', 'fiber tracts-unassigned',
#               'HY-unassigned', 'VMH', 'LPO', 'NLOT', 'AVPV', 'DMH', 'PMv', 'ARH',
#               'PS', 'LSr', 'AP', 'SBPV', 'MS', 'PMd', 'MPN', 'PVpo', 'COAa', 'PVH', 'GPi',
#               'MEA', 'LHA', 'RL', 'VM', 'SAG', 'PoT', 'SPFp', 'VPMpc', 'IO', 'PIL', 'SNr', 'SNc', 'ZI', 'PVHd', 'ECU']
# lregtodrop =['SpC', 'GPi', 'VL-unassigned', 'mfbc', 'ECT', 'SSs', 'LSc', 'VISC', 'EPd', 'IA', 'TT',
#        'ACB', 'epsc', 'cc', 'AIp', 'LSv', 'BMA', 'BST', 'MPO', 'ADP', 'LSv', 'BMA', 'BST', 'MPO', 'ADP', 'fiber tracts-unassigned',
#               'HY-unassigned', 'VMH', 'LPO', 'NLOT', 'AVPV', 'DMH', 'PMv', 'ARH',
#               'PS', 'LSr', 'AP', 'SBPV', 'MS', 'PMd', 'MPN', 'PVpo', 'COAa', 'PVH',
#               'MEA', 'LHA', 'RL', 'VM', 'SAG', 'PoT', 'SPFp', 'VPMpc', 'IO', 'SNr', 'SNc', 'ZI', 'PVHd', 'ECU']
# =============================================================================

forebrain = ["N016-651324", 'N063-709222', 'N067-685221-HS','N068-685221', 'N040-709222', 'AA1521']

lnozero = lnozero.drop('SpC', axis=1)
fnozero = fnozero.drop('SpC', axis=1)
fnozero = fnozero.drop(forebrain, axis=0)
#fnozero = fnozero.drop('SpC', axis=1)
# =============================================================================
# fnoz = fnozero.drop(forebrain, axis=0)
# 
# fnz = fnoz.loc[:, (fnoz != 0).any(axis=0)]
# =============================================================================

#region hindbrain (HB) does not have a volume attached to it, the nodes that are
#getting assigned here
fthresh = 20
lthresh = 150
fns = fnozero.sum(axis=0)
lns = lnozero.sum(axis=0)
fut = fns[fns<fthresh].index.to_list()
lut = lns[lns<lthresh].index.to_list()

#first convert to mm then normalize by region volume
#lnozero['sV'] = lnozero['sV'] + lnozero['sptV']
#lnozero.drop(['sptV', 'HB', 'MY-sat', 'BS', 'MEV', 'sctd', 'brain', 'HEM', 'uf'], axis=1, inplace=True)

#fnozero['sV'] = fnozero['sV'] + fnozero['sptV']
#fnozero.drop('sptV', axis=1, inplace=True)

lnorm = normalize(lnozero, mm=True)
fnorm = normalize(fnozero)

lsum = lnorm.sum(axis=0)
fsum = fnorm.sum(axis=0)


lssort = lsum.sort_values(ascending=False)
fssort = fsum.sort_values(ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].bar(lssort.head(10).index, lssort.head(10).values)
ax[0].set_title('axon length/mm^3')
ax[1].bar(fssort.head(10).index, fssort.head(10).values)
ax[1].set_title('endpoints/mm^3')


def cluster_corr(corr_array, regidx, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to each other 
    
    modified from @jpv88
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    idx_reg = regidx[idx]
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx_reg

fdrop = fnorm.drop(fut, axis=1)

ldrop = lnorm.drop(lut, axis=1)

cef = np.corrcoef(fdrop.T)
lcef = np.corrcoef(ldrop.T)
# %%


lcc, lidx = cluster_corr(lcef, ldrop.T.index)
cov_clust, idxreg = cluster_corr(cef, fdrop.T.index)
# %%


def zoom_heatmap(hmap, idx, start=None, end=None):
    '''
    

    Parameters
    ----------
    hmap : numpy.ndarray
        NxN clustered correlation matrix.
    idx : pandas.Index
        list of indices sorted same as hmap.
    focus : slice
        (optional) desired indices to zoom on.

    Returns
    -------
    pandas.Index, the indices that are being shown if hmap is cut

    '''
    fig, ax = plt.subplots(dpi=300)
    if start:
        labels = idx[start:end]
        
        sns.heatmap(hmap[start:end, start:end], ax=ax, xticklabels=labels, yticklabels=labels)
        return labels
    else:
        labels = idx
        sns.heatmap(hmap, ax=ax)
# %%
#zoom_heatmap(cov_clust, idxreg, start=32, end=50) #14-32 for MB module

# %%
        
zoom_heatmap(lcc, lidx)

zoom_heatmap(cov_clust, idxreg)
# %%
#zoom_heatmap(cov_clust, idxreg, start=12, end=16)
zoom_heatmap(cov_clust, idxreg, start=31,end=57) #43-63 shows vestibular, sensory, medial, 30-40 trigeminal/PB
#zoom_heatmap(cov_clust, idxreg, start=14, end=27) #sensory/autonomic
# %%

#zoom_heatmap(cov_clust, idxreg, start=25, end=35)
#zoom_heatmap(cov_clust, idxreg, start=81, end=95)
# =============================================================================
# hylabels = zoom_heatmap(cov_clust, idxreg, start=165, end=193)
# 
# mblabels = zoom_heatmap(cov_clust, idxreg, start=1, end=36)
# 
# cblabels = zoom_heatmap(cov_clust, idxreg, start=35, end=50)
# 
# randlabels = zoom_heatmap(cov_clust, idxreg, start=112, end=118)
# 
# zoom_heatmap(cov_clust, idxreg, start=117, end=145)
# 
# n040labels = zoom_heatmap(cov_clust, idxreg, start=132, end=138)
# 
# ctplabels = zoom_heatmap(cov_clust, idxreg, start=151, end=170)
# =============================================================================

