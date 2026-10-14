# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:26:10 2026
annotate and resave swc files with region annotation
@author: samkr
"""

from reconstructions.utils.load_data import annotate_swc
from reconstructions.utils.filedirs import celldir, annotated
from tqdm import tqdm
import os

for file in tqdm(os.listdir(celldir), desc='annotating swcs'):
    
    fullfile = os.path.join(celldir, file)
    
    annot_df = annotate_swc(fullfile)

    cellname = file.split('.')[0]
    
    sf = os.path.join(annotated, cellname+'.swc')
    
    annot_df.to_csv(sf, sep=r' ', header=False, index=False)