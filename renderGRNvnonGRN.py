# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:40:58 2026

@author: samkr
"""

from brainrender import Scene, settings
from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils import cameras
import pickle
from tqdm import tqdm
import os

THRESH = 2

mossys = ['AA0922', 'AA1263', 'N010-651324', 'N013-703070', 'N016-715345-HD', 'N017-703070', 'N017-715345-YV', 'N022-703070',
          'N023-715346-PC', 'N024-715345-SA', 'N030-651895', 'N031-651895', 'N031-715345-DS', 'N035-674191-FMR', 'N037-674185-IB',
          'N038-674185', 'N041-674191-AR', 'N044-674191-SP', 'N056-686955-JN', 'N057-686955-SA', 'N113-708369-JN', 
          'N114-708369-HS', 'N115-708369-BP']

frequencies = pickle.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(frequencies)

celldir = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson"

ccf_scene = Scene(atlas_name='allen_mouse_10um')
settings.SHOW_AXES=False
settings.ROOT_COLOR=[0.8,0.8,0.8]
root = ccf_scene.get_actors()[0]
root._needs_silhouette=False

ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.2)
ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.2, color='pink')
ccf_scene.add_brain_region('GRN', silhouette=False, alpha=0.2, color='green')
ccf_scene.add_brain_region('MRN', silhouette=False, alpha=0.2, color='purple')

GRNcells = []
nonGRNcells = []

for cell, targets in merged.iterrows():
    if targets['GRN'] > THRESH:
        GRNcells.append(cell)
    if targets['GRN'] <= THRESH:
        nonGRNcells.append(cell)
        
for file in tqdm(os.listdir(celldir), desc='Loading neurons'):
    cellname = file.split('.')[0]
    filepath = os.path.join(celldir, file)
    if cellname in mossys:
        continue
    if cellname in GRNcells:
        actors = pp.swap_for_brainrender(filepath, axon='green', skip_dendrite=True, soma='green', neurite_radius=4, soma_radius=4)
        for actor in actors:
            ccf_scene.add(actor)
    if cellname in nonGRNcells:
        actors = pp.swap_for_brainrender(filepath, axon='blue', skip_dendrite=True, soma='blue', neurite_radius=4, soma_radius=4)
        for actor in actors:
            ccf_scene.add(actor)
ccf_scene.render(camera=cameras.topcam)