# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:04:02 2026
top reg, corrcoeff, stats about recons
@author: samkr
"""

from reconstructions.utils import preprocess_funcs as pp
import pickle
from reconstructions.utils.filedirs import frequenciespkl, lengthspkl
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'arial'

freqs = pickle.load(open(frequenciespkl, 'rb')).T
lengths = pickle.load(open(lengthspkl, 'rb')).T

lmerged = pp.merge_regions(lengths)
fmerged = pp.merge_regions(freqs)
lnozero = lmerged.loc[:, (lmerged != 0).any(axis=0)]
fnozero = fmerged.loc[:, (fmerged != 0).any(axis=0)]

lpp = pp.preprocess(lnozero, pct=True)
fpp = pp.preprocess(fnozero, pct=True)

lsum = lpp.sum(axis=1)
fsum = fpp.sum(axis=1)

#plot top projected regions by percentage of total dataset
fig, (lax, fax) = plt.subplots(1, 2)
#lax.bar()