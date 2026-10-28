# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:50:11 2026
check axon start parent types
@author: samkr
"""

from reconstructions.utils.filedirs import celldir_forneurom
import pandas as pd
from tqdm import tqdm
import os

for file in tqdm(os.listdir(celldir_forneurom), desc='checking axon start parents'):
    fullfile = os.path.join(celldir_forneurom, file)
    cellname = file.split('.')[0]
    df = pd.read_csv(fullfile, sep=' ', names=['nodeID','structureIdentifier','x','y','z','radius','parent'],
                     comment='#')
    axonstartparent = df.loc[df['structureIdentifier']==2].iloc[0]['parent']
    parentType = df.loc[df['nodeID']==axonstartparent]['structureIdentifier']
    if parentType.values[0] == 3:
        string = 'dendrite'
        print(f'axon comes from {string} for {cellname}')