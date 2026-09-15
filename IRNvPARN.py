# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:17:05 2026
IRN v PARN
@author: samkr
"""

from reconstructions.utils.filedirs import allcoordswapped, frequenciespkl, lengthspkl, parcellation_mappkl, allen_ccf_10um
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.load_data import load_neurons, get_allen_region
import numpy as np
import pandas as pd
import pickle
import nibabel as nib
# %%
def normalize(s):
    sm = np.sum(s)
    sdiv = s/sm
    return sdiv

# %%

allen_ccf = nib.load(allen_ccf_10um)
allen_ccf_data = np.asanyarray(allen_ccf.dataobj)

parcellation_map = pickle.load(open(parcellation_mappkl, 'rb'))

freqs = pickle.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(freqs)
cells = load_neurons(allcoordswapped)

# %%
IRN = []
PARN = []
IRNsoma = {}
PARNsoma = {}
other = []

forebrain = ["N016-651324", 'N063-709222', 'N067-685221-HS','N068-685221', 'N040-709222', 'AA1521']

for name, cell in cells.items():
# =============================================================================
#     if name in forebrain:
#         continue
# =============================================================================
    
    soma = cell['soma']
    coords = [soma['x'], soma['y'], soma['z']]
    tenmicron = np.round([x/10 for x in coords]).astype(int).tolist()
    allenid = allen_ccf_data[tenmicron[0], tenmicron[1], tenmicron[2]]
    parcels = parcellation_map[parcellation_map['parcellation_index'] == allenid]
    region = get_allen_region('structure', parcels)
    if region == 'IRN':
        IRN.append(name)
        IRNsoma[name] = region
    elif region == 'PARN':
        PARN.append(name)
        PARNsoma[name] = region
    else:
        other.append((name, region))
# %%

thresh = 20

summed = merged.sum(axis=0)
todrop = summed[summed < thresh].index
merged = merged.drop(columns=todrop)

# control for cell size
merged_norm = merged.div(merged.sum(axis=1), axis=0)

mn_mean = merged_norm.mean(axis=0)

# mean % endpoints to each region
IRNser = merged_norm.loc[IRN].mean(axis=0)
PARNser = merged_norm.loc[PARN].mean(axis=0)

RNind = pd.Series(index=IRNser.index, dtype=float)

for reg in IRNser.index:
    s = mn_mean[reg]#IRNser[reg] + PARNser[reg]
    if s == 0:
        RNind[reg] = 0
        continue
    i = IRNser[reg] / s
    p = PARNser[reg] / s
    RNind[reg] = i - p  # + = IRN biased, - = PARN biased, 0 = equal

# %%

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'arial'

fig, ax = plt.subplots()

ax.hist(RNind)
ax.set_xticks(np.arange(-2,2.5, 0.5))
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('RN index')
ax.set_ylabel('# of regions')

# %%
'''
writing a plot to check a bunch of different thresholds for endpoints to see if theres a spot where the regions dropped level off
'''
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'arial'

m2 = pp.merge_regions(freqs)

T = np.arange(1,100,10)
reg = []
for t in T:
    sd = m2.sum(axis=0)
    td = sd[sd<t].index
    mt = m2.drop(columns=td)
    reg.append(len(mt.columns))
    
fig2, ax2 = plt.subplots()

ax2.plot(T, reg, '-k')
ax2.set_xlabel('Threshold (# endpoints)')
ax2.set_ylabel('# of regions > thresh')
# %%
from brainrender import Scene, settings
import os
from tqdm import tqdm
from reconstructions.utils import cameras


ccf_scene = Scene(atlas_name='allen_mouse_10um')

settings.SHOW_AXES = False

root = ccf_scene.get_actors()[0]
root._needs_silhouette = False

ccf_scene.add_brain_region('IRN', color='green', alpha=0.2, silhouette=False)
ccf_scene.add_brain_region('PARN', color='purple', alpha=0.2, silhouette=False)

celldir = r"reconstructions\data\IRNPARN_cells\swcsfromjson"

IRNneurons = []
PARNneurons = []

for file in tqdm(os.listdir(celldir), desc='separating IRN v PARN'):
    cellname = file.split('.')[0]
    if cellname in IRN:
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon='green', 
                                         neurite_radius=10, skip_dendrite=True)
        IRNneurons.append(actors)
    if cellname in PARN:
        actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon='purple', 
                                         neurite_radius=10, skip_dendrite=True)
        PARNneurons.append(actors)
        
for actors in IRNneurons:
    for actor in actors:
        ccf_scene.add(actor)
        
for actors in PARNneurons:
    for actor in actors:
        ccf_scene.add(actor)
        
ccf_scene.render(camera=cameras.corcam)