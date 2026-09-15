# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:45:26 2026
read IRN/PARN merfish/scRNAseq files, get cell types that express a given gene
@author: economolab
"""

from reconstructions.utils import filedirs, ABC_utils
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#set paths
scRNAseq_raw = os.path.join(filedirs.local_data_dir, "IRN-PARN-scRNAseq-raw.npy")
scRNAseq_meta = os.path.join(filedirs.local_data_dir, "IRN-PARN-scRNAseq-meta.csv")
MERFISH_freqs = os.path.join(filedirs.local_data_dir, "IRN-PARN-MERFISH-freqs.pkl")

#load scRNAseq data and combine
scRNAseq_genes = ABC_utils.load_gene('scRNAseq')
scRNAseq_raw = np.load(open(scRNAseq_raw, 'rb'))
scRNAseq_meta = pd.read_csv(scRNAseq_meta)

scRNAseq_all = pd.DataFrame(scRNAseq_raw, columns=scRNAseq_genes, index=scRNAseq_meta.index)

#%%
#genes of interest
GOI = ['Calb2', 'Cartpt', 'Crh', 'Dbh', 'Gal', 'Grp', 'Ntrk1', 'Sim1', 'Sst', 'Slc32a1', 'Slc17a7', 'Slc17a6']
scRNAseq_goi = scRNAseq_all[GOI]

#%%
#plot some metrics about data
def plot_metric(scRNAseq, metric):
    match metric:
        case 'mean':
            def mean_in_positive(scRNAseq):
                means = pd.Series(data=np.zeros(len(scRNAseq.columns)), index=scRNAseq.columns)
                for gene in scRNAseq:
                    vals = scRNAseq[gene]
                    nonzero = vals[vals != 0]
                    mean = nonzero.mean()
                    means[gene] = mean
                return means
            
            counts = mean_in_positive(scRNAseq)
            sf = os.path.join(filedirs.plotdir, 'meaninpos_scRNAseq_counts.png')
            title = 'Mean # counts in positive cells in scRNAseq data'
        case 'max':
            counts = scRNAseq.max(axis=0)
            sf = os.path.join(filedirs.plotdir, 'max_scRNAseq_counts.png')
            title = 'Max # counts in scRNAseq data'
            
    fig, ax = plt.subplots()
    
    ax.bar(counts.index, counts)
    ax.tick_params(rotation=90, axis='x')
    ax.set_ylabel('# counts')
    ax.set_xlabel('Gene')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(sf)
    return counts
#%%
max_counts = plot_metric(scRNAseq_goi, metric='max')
mean_counts = plot_metric(scRNAseq_goi, metric='mean')

#%%
#plot distributions of types for cells that express a given gene in scRNAseq data
thresh = 0
for gene in GOI:
    

#savedir for plots
scsave = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\plots\types_in_genes'

#load MERFISH data
MERFISH_freqs = pickle.load(open(MERFISH_freqs, 'rb'))


