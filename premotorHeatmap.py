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

thresh=10

frequencies = pickle.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(frequencies)

premotor = merged.loc[(merged['XII']>thresh) | (merged['VII']>thresh) | (merged['V']>thresh)]

premotorMNs = premotor[['XII', 'V', 'VII']]

