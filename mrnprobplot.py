# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:08:39 2026
make P(MRN|GRN) v P(MRN|~GRN)
@author: samkr
"""
from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import preprocess_funcs as pp
import pickle

THRESH = 2

frequencies = pickle.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(frequencies)

