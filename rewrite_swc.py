# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:07:40 2026
replace 5s and 6s structureID in swc for analysis by neurom
5 and 6 were used in some iterations of allen reconstructions to mark branch
and endpoints, have moved away now but some will still have this
meaning that these nodes may be ignored depending how analysis is done
@author: samkr
"""

import os
import pandas as pd
from reconstructions.utils.filedirs import celldir, celldir_forneurom
from tqdm import tqdm
import numpy as np
from collections import defaultdict

#%%
def collapse_duplicates(df, tol=1e-6):
    '''
    checks for duplicate nodes, and if it finds any, reassigns the parent of the duplicate node
    to any of the duplicate nodes children
    the issue related to this script is that nodes are trifurcating, while some measurements of neurom
    (bifurcation_partitions assume bifurcating nodes? actually as i think about this i shouldn't need to do this 
     since I will create trifurcating nodes by doing this, maybe it just doesn't like duplicate nodes?)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame created by reading in an .swc file.
    tol : float, optional
        tolerance to count a node as a duplicate. The default is 1e-6 (microns).

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    df = df.copy()
    xyz = df[['x','y','z']].astype(float)
    drop = []
    for node, row in tqdm(df.iterrows(), desc='Removing dupe nodes'):
        p = row['parentNumber']
        if p == -1 or p not in df.index or df.loc[p, 'structureIdentifier'] == 1:
            continue
        #duplicate nodes have the exact same coordinates
        if xyz.loc[node].values.sum() == xyz.loc[p].values.sum():
            df.loc[df['parentNumber'] == node, 'parentNumber'] = p
            drop.append(node)
    return df.drop(index=drop)

colnames = ['sampleNumber', 'structureIdentifier', 'x', 'y', 'z', 'radius', 'parentNumber']

n006 = pd.read_csv(r'C:/Users/samkr/OneDrive/Documents/GitHub/Reconstruction_code/reconstructions/data/IRNPARN_cells/swcsfromjson/N006-703070.swc', sep=' ', comment='#', names=colnames, index_col=['sampleNumber'])

n006 = collapse_duplicates(n006)
#%%

sdir = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swc_for_neurom_v2"

colnames = ['sampleNumber', 'structureIdentifier', 'x', 'y', 'z', 'radius', 'parentNumber']

cellnames = []
track_modified = np.zeros(len(os.listdir(celldir)))

for i, file in tqdm(enumerate(os.listdir(celldir)), desc='Replacing 5|6'):
    fullfile = os.path.join(celldir, file)
    cellname = file.split('.')[0]
    cellnames.append(cellname)
    celldf = pd.read_csv(fullfile, sep=' ', comment='#', names=colnames, index_col=['sampleNumber'])
    copycelldf = celldf.copy()
    for sampleNumber, vals in celldf.iterrows():
        if vals['structureIdentifier'] in [1,2,3]:
            continue
        if vals['structureIdentifier'] in [5,6]:
            track_modified[i] = 1
            parent = vals['parentNumber']
            parent_compartment = celldf.loc[int(parent)]['structureIdentifier']
            while parent_compartment not in (2, 3):
                parent = celldf.loc[int(parent)]['parentNumber']
                parent_compartment = celldf.loc[int(parent)]['structureIdentifier']

            copycelldf.loc[sampleNumber, 'structureIdentifier'] = parent_compartment
            
    #copycelldf = collapse_duplicates(copycelldf)

    copycelldf.to_csv(os.path.join(sdir, cellname+'.swc'), sep=' ', header=False, index=True)

#%%

for val, name in zip(track_modified, cellnames):
    if val != 0:
        print(name)

errors = defaultdict(list)

for file in tqdm(os.listdir(sdir), desc='Verifying replacements'):
    fullfile = os.path.join(sdir, file)
    cellname = file.split('.')[0]
    celldf = pd.read_csv(fullfile, sep=' ', names=colnames, index_col=['sampleNumber'])
    to_check = celldf['structureIdentifier']
    for i, sid in enumerate(to_check):
        if sid not in (1,2,3):
            errors[cellname].append(i)

print(errors)

#%%
import os
import yaml
import traceback
import neurom as nm

swcdir = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swc_for_neurom_v2"
cfg = yaml.safe_load(open(r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\config_for_neuromstats.txt"))

for f in tqdm(os.listdir(swcdir),desc='testing measurements'):
    morph = nm.load_morphology(os.path.join(swcdir, f))
    for feat in cfg['neurite']:
        for nt in cfg['neurite_type']:
            try:
                nm.get(feat, morph, neurite_type=getattr(nm.NeuriteType, nt.lower()))
            except Exception as e:
                print(f"{f} | {feat} | {nt} -> {type(e).__name__}: {e}")
                        
