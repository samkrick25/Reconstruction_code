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
from reconstructions.utils.cellLists import cells_by_pheno
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

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

structure_vols = {}
with open(ccf_vols3, 'r') as svols:
    reader = csv.reader(svols)
    for row in reader:
        structure_vols[row[0]] = np.float64(row[1])
        
# %%
# could log scale this data to try to reduce effects of IRN/PARN being so highly targeted? idk its worth a thought
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.filedirs import lengthspkl, ccf_vols3, structure_ont_info
structure_to_ont = pickle.load(open(structure_ont_info, 'rb'))
freqs = pickle.load(open(r'reconstructions\data\freqs_hopefullygood.pkl', 'rb')).T
lengths = pickle.load(open(lengthspkl, 'rb')).T
lmerged = pp.merge_regions(lengths)
fmerged = pp.merge_regions(freqs)
lnozero = lmerged.loc[:, (lmerged != 0).any(axis=0)]
fnozero = fmerged.loc[:, (fmerged != 0).any(axis=0)]

#dorp a duplicate cell, will be fixed whenever I get new ones and rerun 
#the projection calculations
lnozero = lnozero.drop(index=['N044-674191'])

#pp.write_targeted_regions_to_excel(freqs, r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\targeted_reg_lat.xlsx')

#%%
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



def show_heatmap(hmap, idx, start=None, end=None):
    '''
    show a heatmap of clustered correlations, optionally zoom in on certain indices

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
        if start == 0:
            labels = idx[:end]
            
            sns.heatmap(hmap[start:end, start:end], ax=ax, xticklabels=labels, yticklabels=labels)
            return labels
        if start > 0:
            labels = idx[start:end]
            
            sns.heatmap(hmap[start:end, start:end], ax=ax, xticklabels=labels, yticklabels=labels)
            return labels
    else:
        labels = idx
        sns.heatmap(hmap, ax=ax)

import os

def do_pca_and_plot(data, n_components=None, sd=None, plot=False):
    '''
    do pca on your data, returns pca object as well as transformed data, also plots loadings onto the top 10 pcs

    Parameters
    ----------
    data : array-like
        data to do PCA on
    sf : str
        desired save location, directory
    n_components : int
        desired number of components for PCA to find
    Returns
    -------
    PCA object and transformed data.

    '''
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    
    if n_components:
        pca=PCA(n_components=n_components)
    else:
        pca=PCA()
    
    pca.fit(scaled)
    scaled = pca.fit_transform(scaled)
    

    loadings = pd.DataFrame(pca.components_, columns = data.columns, index = [f'PC{i+1}' for i in range(pca.n_components_)])

    def get_top_loading(df, component, n=15):
        pc = df.loc[component]
        
        return pc.reindex(pc.abs().sort_values(ascending=False).index).head(n)
    
    if plot:
        for i in range(1,n_components+1 if n_components else 11):
            string = 'PC'+str(i)
            l = get_top_loading(loadings, string)
            plt.bar(l.index, l.values)
            plt.title('Loadings onto '+string)
            plt.tick_params(axis='x', labelrotation=90)
            plt.tight_layout()
            sf = os.path.join(sd,string+'.png')
            plt.savefig(sf)
            plt.close()
            
    return pca, scaled, loadings

from sklearn.decomposition import SparsePCA

def do_spca_and_plot(data, alpha, n_components=None, sd=None, plot=False):
    '''
    do pca on your data, returns pca object as well as transformed data, also plots loadings onto the top 10 pcs

    Parameters
    ----------
    data : pd.DataFrame
        data to do PCA on
    sf : str
        desired save location, directory
    n_components : int
        desired number of components for PCA to find
    Returns
    -------
    PCA object and transformed data stored as DataFrame.

    '''
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    
    if n_components:
        pca=SparsePCA(alpha = alpha, n_components=n_components)
    else:
        pca=SparsePCA(alpha=alpha)
    
    pca.fit(scaled)
    scaled = pca.fit_transform(scaled)
    
    scaled = pd.DataFrame(data=scaled, index=data.index)
    
    loadings = pd.DataFrame(pca.components_, columns = data.columns, index = [f'PC{i+1}' for i in range(pca.n_components_)])

    def get_top_loading(df, component, n=15):
        pc = df.loc[component]
        
        return pc.reindex(pc.abs().sort_values(ascending=False).index).head(n)

    if plot:
        for i in range(1,n_components+1):
            string = 'PC'+str(i)
            l = get_top_loading(loadings, string)
            plt.bar(l.index, l.values)
            plt.title('Loadings onto '+string)
            plt.tick_params(axis='x', labelrotation=90)
            plt.tight_layout()
            sf = os.path.join(sd,string+'.png')
            plt.savefig(sf)
            plt.close()
        
    return pca, scaled


import umap
from reconstructions.utils.cellLists import cells_by_pheno, pColors

def umap_and_plot(pcadata, index, sd=None, **kwargs):
    '''
    UMAP data and scatter plot

    Parameters
    ----------
    pcadata : ndarray
        data already dimensionality reduced by PCA.
    index : pd.Index
        index of the data you are plotting
    sd : str
        path that you want to save UMAP plot to, default None/not saved
    **kwargs : kwargs
        keyword arguments to pass to UMAP fit_transform().

    Returns
    -------
    UMAP embedding DataFrame.

    '''
    
    reducer = umap.UMAP(**kwargs)

    scaled_sPCA_forUMAP = StandardScaler().fit_transform(pcadata)

    embedding = reducer.fit_transform(scaled_sPCA_forUMAP)
    
    embeddingdf = pd.DataFrame(embedding, index=index, columns = ['UMAP1', 'UMAP2'])

    color_map = {}
    for cell in embeddingdf.index:
        pheno = cells_to_pheno[cell]
        color = pColors[pheno]
        color_map[cell] = color

    colors = [val for key, val in color_map.items()]


    plt.scatter(embeddingdf['UMAP1'], embeddingdf['UMAP2'], c=colors, alpha=0.7)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.xticks([])
    plt.yticks([])
    if sd:
        plt.savefig(sd)
    
    return embeddingdf


from reconstructions.utils.cellLists import cells_by_pheno

#create a mapping from cell name to pheno identity
def map_val_to_key(d):
    nd = {}
    for k, v in d.items():
        for val in v:
            nd[val]=k
    return nd

cells_to_pheno = map_val_to_key(cells_by_pheno)

def get_phenos(cellnames):
    phenodict = {'mossy': 0, 'sensory': 0 , 'premotor':0, 'forebrain':0, 'sensorimotor':0, 'vestibular':0, 'GRN':0}
    for cell in cellnames:
        pheno = cells_to_pheno[cell]
        phenodict[pheno] += 1

    return phenodict

#%% lists of regions to iteratively drop as I group them

#ok start, still some stuff i should group -> ZI doesn't cluster with the rest of HY stuff since it sometimes pops up in MB cells
#think I should throw some smaller regions together idk?
HY = ['AVPV', 'LPO', 'ADP', 'VMH', 'MPO', 'DMH', 'AP', 'fiber tracts-unassigned',
       'PVH', 'PVpo', 'ARH', 'MPN', 'AHN', 'BAC', 'SBPV', 'COAa', 'PS', 'NLOT',
       'PD', 'HY-unassigned', 'PMv', 'In', 'MS', 'PMd', 'NDB', 'PST', 'PVHd',
       'PH', 'RL', 'PVi', 'MEA', 'IIn', 'LHA', 'RCH', 'PVa', ]

CTXsp = ['LSv', 'STR-unassigned', 'mfbc', 'CTXsp-unassigned', 'BMA', 'BST', 'LSr']

CTX = ['RR', 'OLF-unassigned', 'ACB', 'LA', 'TT', 'GPi', 'SI', 'GPe', 'IA',
       'epsc', 'BLA', 'cc', 'VL-unassigned', 'SSs', 'ECT', 'AIp', 'CEA', 'SH',
       'VISC', 'LSc', 'EPd', 'PAL-unassigned', 'DP']

CB = ['IP', 'CB', 'UVU', 'CUL', 'DEC', 'cbc', 'PFL', 'PRM', 'PYR', 'y', 'arb',
       'DN', 'VeCB', 'CENT', 'ICB', 'VCO', 'NOD', 'CB-unassigned', 'FL', 'V4r',
       'FN', 'DCO', 'cbp', 'FOTU', 'SUV', 'chpl', 'AN', 'COPY', 'SIM']

MB = ['Eth', 'RPF', 'RN', 'III', 'Pa4', 'IIIn', 'MA3', 'RH', 'PAG', 'PRNr',
       'NPC', 'brain-unassigned', 'MRN', 'MH', 'MB-unassigned', 'PRNc',
       'SPFm', 'LH', 'V3-unassigned', 'APN', 'mfsbshy', 'SCm', 'PPN', 'CUN']

TH = ['VM', 'PF', 'VPL', 'VPM', 'VPMpc', 'VPLpc', 'PCN', 'PO', 'VAL',
       'TH-unassigned', 'cst', 'PVT', 'MD', 'IAM', 'CM']

vestibular = ['MV', 'LAV', 'SPIV']

to_merge = {'TH':TH, 'MB':MB, 'CB':CB, 'CTX':CTX, 'CTXsp':CTXsp, 'HY':HY, 'vestibular':vestibular}

#%% region by region, using to decide how to group regions by looking at what stuff is highly correlated
#first not normalized, including spinal cord thing

l_notnorm_corr = np.corrcoef(lnozero.T)

no_norm_clusters, no_norm_idx = cluster_corr(l_notnorm_corr, lnozero.columns)

show_heatmap(no_norm_clusters, no_norm_idx)

#%% merge clustered regions

lcomb = lnozero.copy()

for group, regs in to_merge.items():
    groupsums = lnozero[regs].sum(axis=1)
    lcomb[group] = groupsums
    lcomb.drop(regs, axis=1, inplace=True)

#%% recorrelate and cluster on merged regions

comb_corr = np.corrcoef(lcomb.T)

comb_clust, comb_idx = cluster_corr(comb_corr, lcomb.columns)

show_heatmap(comb_clust, comb_idx)

#%% pca on combined stuff so far, cell clustering
sd = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\combining_regions_PCA_loadings_v1\loadings"
comb_pca, comb_scaled, _ = do_pca_and_plot(lcomb, n_components=20, sd=sd, plot=True)

pca_corr = np.corrcoef(lcomb)

comb_clust, comb_idx = cluster_corr(pca_corr, lcomb.index)

show_heatmap(comb_clust, comb_idx)

#%% sparse pca
sd = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\combining_reg_sPCA_v1\loadings"
comb_spca, comb_scaled_sparse = do_spca_and_plot(lcomb, alpha=2, plot=True, sd=sd, n_components=20)

comb_sparse_corr = np.corrcoef(comb_scaled_sparse)

comb_s_clusters, comb_s_idx = cluster_corr(comb_sparse_corr, lcomb.index)

show_heatmap(comb_s_clusters, comb_s_idx)
#%% normalized by region volume
lnozero=lnozero.drop('SpC', axis=1) #drop custom spinal cord annotation since no ccf volume associated
#%%
l_norm = normalize(lnozero, mm=True)

lnorm_corr = np.corrcoef(l_norm.T)

lnorm_clusters, lnorm_idx = cluster_corr(lnorm_corr, l_norm.columns)

show_heatmap(lnorm_clusters, lnorm_idx)
#%% cell by cell correlations
 
lnozero = lnozero.drop('SpC', axis=1)
fnozero = fnozero.drop('SpC', axis=1)
#%%
#mm converts from microns to mm
lnorm = normalize(lnozero, mm=True)
#this will have endpoint frequency of cells in reg in ends/um^3
fnorm = normalize(fnozero)

#%% cluster cells based on freq

cellcef = np.corrcoef(fnorm)
cellclust, cellidx = cluster_corr(cellcef, fnorm.index)

#%% plot freq

show_heatmap(cellclust, cellidx)

#%% cluster cells based on length

cellcefL = np.corrcoef(lnorm)
cellclustL, cellidxL = cluster_corr(cellcefL, lnorm.index)

#%% plot length

show_heatmap(cellclustL, cellidxL)

#%% plot with mossys removed and log scale data
#log scale
llog = np.log1p(lnorm)

#%% drop mossys
llog = lnorm.drop(cells_by_pheno['mossy'], axis=0)

#%% crosscorrelate

cc_nomossy = np.corrcoef(llog)
clust_nomossy, idx_nomossy = cluster_corr(cc_nomossy, llog.index)

#%% plot

show_heatmap(clust_nomossy, llog.index)

#%% cut forebrain projecting and mossy fiber cells, they are super distinct already
#and a lot of the high variance PCs will be explained by them i believe
#probably vestibular as well

lnozero = lnozero.drop(cells_by_pheno['forebrain'], axis=0)
lnozero = lnozero.drop(cells_by_pheno['vestibular'], axis=0)

#lnozero = lnozero.drop(cells_by_pheno[''])

#%% find mean proj strength for region from the cells that project there
#goal is to filter out regions that just recieve really light projections/might be vestigial
#but keep regions that are targeted rarely but well innervated when they are targeted

means = pd.Series(index=lnorm.columns)

for reg, vals in lnorm.items():
    v = vals[vals != 0]
    means[reg] = v.sum()/len(v)
#%% find threshold for dropping regions that recieve little meaningful innervation

ts2 = np.arange(0, 0.2, 0.001)
regcounts2 = []
for t in ts2:
    below = means[means<t].index
    regcounts2.append(len(below))

plt.plot(ts2, regcounts2)

#%%

lmean = lnorm.mean(axis=0)
lsum = lnorm.sum(axis=0)

#%% drop regions that recieve innervation below a certain threshold
#ill do a few different things, try on a sum threshold, mean threshold,
#mean over cells that project to a given region threshold
sthresh = 1
todropsum = lsum[lsum<sthresh].index
l_sum = lnorm.drop(columns=todropsum)

mthresh = 0.005
todropm = lmean[lmean<mthresh].index
l_mean = lnorm.drop(columns=todropm)

mthresh2 = 0.06
todropm2 = means[means<mthresh2].index
l_mean2 = lnorm.drop(columns=todropm2)


    
#%% preprocess data, keep all regions except my custom spinal cord reg since I can't volume normalize that

lnozero = lnozero.drop('SpC', axis=1)
#%% normalize length by region volume
lnorm = normalize(lnozero, mm=True)

#%% scale and fit data

sd = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\allregs_volnorm_length_PCA\loadings"

pca, scaled_PCA, loadings = do_pca_and_plot(lnorm, sd, n_components=20, plot=False)

#%%
correlation = np.corrcoef(scaled_PCA)

corrmat, index = cluster_corr(correlation, lnorm.index)

show_heatmap(corrmat, index)

#%% test alphas for sparse PCA
from sklearn.utils import resample
from sklearn.decomposition import SparsePCA
from tqdm import tqdm

def component_stability(data, alpha, n_components, n_boots=20):
    all_components = []
    
    for _ in tqdm(range(n_boots), desc='bootstrapping and finding sparse PCs'):
        sample = resample(data, n_samples=len(data), random_state=None)
        spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
        spca.fit(sample)
        all_components.append(spca.components_)
    
    # Measure variance of loadings across bootstrap runs
    all_components = np.array(all_components)  # (n_boots, n_components, n_features)
    stability = 1 - np.mean(np.std(all_components, axis=0))
    return stability

alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

stabilities = [component_stability(l_mean, a, n_components=20) for a in alphas]

plt.plot(alphas, stabilities, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Stability (higher = better)')
plt.xscale('log')
plt.title('Component Stability vs Alpha')
plt.show()


#%% plot top sparse pcs
sd= r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\allregs_volnorm_length_sPCA\loadings'
spca, scaled_sPCA = do_spca_and_plot(lnorm, plot=True, sd=sd, alpha=2, n_components=20)

#%%
sPCAcor = np.corrcoef(scaled_sPCA)

sPCAclust, sPCAidx = cluster_corr(sPCAcor, lnorm.index)

show_heatmap(sPCAclust, sPCAidx)

#%% pca, crosscorrelate, cluster for sum thresholded data
sd = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\mean_bytargetingcells_threshold_sPCA\loadings'
pca, scaled_thresh = do_spca_and_plot(l_sum, alpha=2, plot=False, sd=sd, n_components=20)

corrs = np.corrcoef(scaled_thresh)

clust, idx = cluster_corr(corrs, l_mean2.index)

show_heatmap(clust, idx)

#so far what this is telling me is that I do need more than just targeted regions to cluster these cells well
#again, most other papers doing this kind of thing look at cortical neurons, one on ascending spinal projections which 
#should be fairly topographic, but this is another problem of medulla being medulla


#%% umap stuff

sd= r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\allregs_volnorm_length_sPCA\umap3.png'
sPCA_embeddings_nothresh = umap_and_plot(scaled_sPCA, lnorm.index, sd)

#%%
sd = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\sumthresholded_sPCA\umap.png'
sPCA_meanovertargeting_UMAPembeddings = umap_and_plot(scaled_thresh, lnorm.index, sd)

#%% umap PCA

sd = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\allregs_volnorm_length_PCA\umap.png'

PCA_embeddings = umap_and_plot(scaled_PCA, lnorm.index, sd)

#%%

sd = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\plots\PCA\notnorm_sPCA\loadings'

pca, scaled_notnorm = do_spca_and_plot(lnozero, alpha=2, n_components=20, sd=sd, plot=True)
#%%
corrs = np.corrcoef(scaled_notnorm)

clust, idx = cluster_corr(corrs, lnozero.index)

show_heatmap(clust, idx)