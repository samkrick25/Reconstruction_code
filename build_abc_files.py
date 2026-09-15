# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:14:59 2026
abc atlas stuff for reconstructions, this is using ABC environment b/c of dependencies
@author: economolab
"""

from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache as apc
import pandas as pd
import numpy as np
from reconstructions.utils import filedirs
import os


gene_names = ['Gal', 'Cartpt', 'Grp', 'Dbh', 'Ntrk1', 'Slc17a6', 'Slc32a1', 'Crh', 'Sst', 'Sim1']


def build_10x_files():
    gene10x = pd.read_csv(r"D:\allen_brain_atlas\metadata\WMB-10X\20241115\gene.csv").set_index('gene_identifier')
    
    glist10x = gene10x['gene_symbol'].values
    
    sf = os.path.join(filedirs.abc_local, 'scRNAseq_genes.npy')

    np.save(sf, glist10x)
    
def build_MERFISH_files():
    geneMERFISH = pd.read_csv(r"D:\allen_brain_atlas\metadata\MERFISH-C57BL6J-638850\20241115\gene.csv").set_index('gene_identifier')
    
    glistMERFISH = geneMERFISH['gene_symbol'].values
    
    sf = os.path.join(filedirs.abc_local, 'MERFISH_genes.npy')
    
    np.save(sf, glistMERFISH)
    
build_10x_files()
build_MERFISH_files()