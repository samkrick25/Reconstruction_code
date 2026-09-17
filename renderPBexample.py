# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:56:43 2026
show PB and some cells in it
@author: samkr
"""

from reconstructions.utils.filedirs import celldir
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.cellLists import cells_by_pheno
from reconstructions.utils import cameras as c
from brainrender import Scene, settings
import os
from tqdm import tqdm

toRender = ['N005-674185', 'N032-674185-IB', 'N086-686955', 'N074-709222', 'AA1313', 'N059-709222', 'AA1521', 'AA1535', 'N067-685221-HS']

settings.INTERACTIVE = False
settings.OFFSCREEN = True
settings.BACKGROUND_COLOR = 'black'
settings.ROOT_ALPHA=0.01
settings.ROOT_COLOR='black'

scene = Scene(atlas_name='allen_mouse_10um')

scene.get_actors()[0]._needs_silhouette = False

scene.add_brain_region('PB', color='white', alpha=0.15, silhouette=False)

PBmesh = scene.get_actors()[1].mesh.vertices

medplane=scene.atlas.get_plane(plane='sagittal',norm=(0,0,1))
scene.slice(plane=medplane)


colors = {'sensory': 'red', 'premotor': 'cyan', 'mossy': 'green', 'GRN': 'purple', 'sensorimotor': 'pink', 'vestibular': 'yellow', 'forebrain': 'orange'}

for file in tqdm(os.listdir(celldir), desc='Adding cells to scene'):
    cellname = file.split(".")[0]
    if cellname in toRender:
        if cellname in cells_by_pheno['sensory']:
            #continue
            actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['sensory'], 
                                             skip_dendrite = True, neurite_radius = 5, soma_radius=0, mesh=PBmesh)
        if cellname in cells_by_pheno['sensorimotor']:
            #continue
            actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['sensorimotor'], 
                                             skip_dendrite = True, neurite_radius = 5, soma_radius=0, mesh=PBmesh)
        if cellname in cells_by_pheno['premotor']:
            #continue
            actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['premotor'], 
                                             skip_dendrite = True, neurite_radius = 5, soma_radius=0, mesh=PBmesh)
        if cellname in cells_by_pheno['GRN']:
            #continue
            actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['GRN'], 
                                             skip_dendrite = True, neurite_radius = 5, soma_radius=0, mesh=PBmesh)
        if cellname in cells_by_pheno['forebrain']:
            #continue
            actors = pp.swap_for_brainrender(os.path.join(celldir, file), axon=colors['forebrain'], 
                                             skip_dendrite = True, neurite_radius = 5, soma_radius=0, mesh=PBmesh)
        
        for actor in actors:
            scene.add(actor)
            
sf = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\images\phenos\examp_cells_in_PB_justPB.png'

scene.screenshot(sf, camera=c.sagcam, scale=6)
        
    