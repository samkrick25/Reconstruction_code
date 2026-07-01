# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:37:33 2026
innervation of regions looking at axon length/mm^3
@author: samkr
"""

from reconstructions.utils.filedirs import lengthspkl, ccf_structure_vols2_mm, frequenciespkl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
from reconstructions.utils import preprocess_funcs as pp
import csv
import numpy as np

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

freqs = pickle.load(open(frequenciespkl, 'rb')).T
lengths = pickle.load(open(lengthspkl, 'rb')).T
lmerged = pp.merge_regions(lengths)
fmerged = pp.merge_regions(freqs)
lnozero = lmerged.loc[:, (lmerged != 0).any(axis=0)]
fnozero = fmerged.loc[:, (fmerged != 0).any(axis=0)]

structure_vols = {}
with open(ccf_structure_vols2_mm, 'r') as svols:
    reader = csv.reader(svols)
    for row in reader:
        structure_vols[row[0]] = np.float64(row[1])

lnozero.drop('SpC', axis=1, inplace=True)
#fnozero.drop('SpC', axis=1, inplace=True)

#first convert to mm then normalize by region volume
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