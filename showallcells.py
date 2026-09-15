# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:48:20 2026
all cell quick visualization
@author: samkr
"""
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.filedirs import celldir
from brainrender import Scene, settings
import os
from tqdm import tqdm
from reconstructions.utils import cameras

settings.INTERACTIVE=False
settings.SHOW_AXES = False
settings.OFFSCREEN=True

scene = Scene(atlas_name='allen_mouse_10um')
scene.get_actors()[0]._needs_silhouette = False

for cell in tqdm(os.listdir(celldir),desc='Loading cells'):
    filename = os.path.join(celldir, cell)
    actors = pp.swap_for_brainrender(filename, axon='black', neurite_radius=5, skip_dendrite=True, soma_radius=10)
    for actor in actors:
        scene.add(actor)
    
    
sf = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\images\allcells_sag.png'
scene.screenshot(sf, camera=cameras.sagcam, scale=6)
