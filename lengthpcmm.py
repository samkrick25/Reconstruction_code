# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:37:33 2026
innervation of regions looking at axon length/mm^3
@author: samkr
"""

from reconstructions.utils.filedirs import lengthspkl, ccf_structure_vols
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
from reconstructions.utils import preprocess_funcs as pp
import csv

lengths = pickle.load(open(lengthspkl, 'rb')).T
merged = pp.merge_regions(lengths)

structure_vols = {}
with open(ccf_structure_vols, 'r') as svols:
    reader = csv.reader(svols)
    for row in reader:
        structure_vols[row[0]] = row[1]

merged = merged.apply(lambda x: x.values/structure_vols[x.get_index()], axis=0)
