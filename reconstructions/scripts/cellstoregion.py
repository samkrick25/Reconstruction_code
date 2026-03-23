# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:25:50 2026
find cells to a given 

@author: samkr
"""
from reconstructions.utils import preprocess_funcs
from reconstructions.utils.filedirs import freqspkl

regionabv='MRN'
poscells, negcells = preprocess_funcs.get_cells_to_region(freqspkl, regionabv)
