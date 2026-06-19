# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:35:10 2026
screenshot premotor cells with the motor nuclei they project to 
@author: samkr
"""

from brainrender import Scene, settings
from reconstructions.utils import preprocess_funcs as pp
import numpy as np
from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import cameras
import pickle
import os
from tqdm import tqdm
from pathlib import Path

THRESH=0

savedir = r'images\premotors'
swcdir = r'reconstructions\data\IRNPARN_cells\swcsfromjson'

frequencies = pickle.load(open(frequenciespkl, 'rb')).T
merged = pp.merge_regions(frequencies)


for cell, row in tqdm(merged.iterrows(), desc='Screenshotting neurons'):

    ccf_scene = Scene(atlas_name='allen_mouse_10um')
    root = ccf_scene.get_actors()[0]
    root._needs_silhouette=False

    ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.2)
    ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.2, color='pink')
    fn = cell+'.swc'

    if cell == 'N004-674185-DS':
        continue
    path = os.path.join(swcdir, fn)
    [xii, v, vii] = [row['XII'], row['V'], row['VII']]
    if xii>THRESH:
        ccf_scene.add_brain_region('XII', color='purple', alpha=0.2, silhouette=False)
    if v>THRESH:
        ccf_scene.add_brain_region('V', silhouette=False, color='green', alpha=0.2)
    if vii>THRESH:
        ccf_scene.add_brain_region('VII', color='blue', alpha=0.2, silhouette=False)
    if sum([xii, v, vii]) > THRESH:
        cellsavedir = os.path.join(savedir, cell)
        Path(cellsavedir).mkdir(parents=True, exist_ok=True)
        actors = pp.swap_for_brainrender(path, neurite_radius=10)
        for actor in actors:
            ccf_scene.add(actor)
        
        ccf_scene.screenshot(name=cellsavedir+'\\'+cell+'_sag.png', camera=cameras.MYsagcam, scale=3)
        ccf_scene.close()