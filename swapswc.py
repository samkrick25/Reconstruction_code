# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:03:41 2026
swap swc coords
@author: samkr
"""
from reconstructions.utils import load_data as ld
import pandas as pd
import os

onecelldir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\premotorsU19\one'
mixcelldir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\premotorsU19\mix'
onesavedir = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\premotorsU19\swap\one"
mixsavedir= r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\premotorsU19\swap\mix"

for file in os.listdir(onecelldir):
    fn = os.path.join(onecelldir, file)
    swc_df = ld.swap_swc(fn)
    swc_df.to_csv(open(onesavedir+'\\'+file, 'w'), sep=' ', index=False, header=False)
    
for file in os.listdir(mixcelldir):
    fn = os.path.join(mixcelldir, file)
    swc_df = ld.swap_swc(fn)
    swc_df.to_csv(open(mixsavedir+'\\'+file, 'w'), sep=' ', index=False, header=False)

