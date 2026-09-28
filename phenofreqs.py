# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:16:40 2026
create frequency dataframes by phenotype
right now only doing premotor stuff and plotting some endpoint things for u19 meeting
but can flesh this script out
@author: samkr
"""

from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils.cellLists import cells_by_pheno
from reconstructions.utils.preprocess_funcs import merge_regions
import pickle
import pandas as pd
import numpy as np

F = pickle.load(open(frequenciespkl, 'rb')).T
F = merge_regions(F)

thresh=0
M = F[F['XII']+F['V']+F['VII'] > thresh]

M_sum = M[['XII', 'VII', 'V']].sum(axis=1)

def get_df(df, key=None):
    return df.loc[cells_by_pheno[key]]

F_premotor = get_df(F, 'premotor')

F_premotor = F_premotor[['XII', 'VII', 'V']]


#%% plots
import matplotlib.pyplot as plt

scatfig, scatax = plt.subplots(figsize=(1,5), dpi=300)
x = np.zeros(len(M_sum.values))
scatax.scatter(x, M_sum.values, s=5, c='k', alpha=0.7)
scatax.tick_params(labelbottom=False)
scatax.set_xticks([])
scatax.set_yscale('log', base=10)
scatax.set_ylabel('log(# of endpoints in motor nuclei)')
scatax.set_xlabel('Cells')
xF = np.zeros(len(F_premotor))
fdots = scatax.scatter(xF, F_premotor.sum(axis=1).values, c='cyan', s=5, label='premotor phenotype')
#scatax.legend(handles=[fdots], loc='upper right')
#scatfig.tight_layout()
