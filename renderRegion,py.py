# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:33:43 2026

@author: samkr
"""

from brainrender import Scene, settings
from reconstructions.utils.preprocess_funcs import swap_for_brainrender as sfb
from reconstructions.utils import cellLists

celldir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson'

settings.ROOT_ALPHA = 0.1

ccf_scene = Scene(atlas_name="allen_mouse_10um")
root = ccf_scene.get_actors()[0]
root._needs_silhouette=False



# =============================================================================
# regtoadd = ['IRN', 'PARN']
# for reg in regtoadd:
#     ccf_scene.add_brain_region(reg, silhouette=False, color='pink', alpha=0.25)
# =============================================================================
c = ['orange', 'blue']
PB = ['V', 'PSV']
for reg, color in zip(PB, c):
    ccf_scene.add_brain_region(reg, silhouette=False, color=color, alpha=0.2)

# =============================================================================
# medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
# ccf_scene.slice(plane=medplane)
# 
# =============================================================================
cells = [
         r"reconstructions\data\IRNPARN_cells\swcsfromjson\N058-709222.swc",
         r"reconstructions\data\IRNPARN_cells\swcsfromjson\AA0947.swc"
         ]

cell =  r"reconstructions\data\IRNPARN_cells\swcsfromjson\N003-703070.swc"

import os
from tqdm import tqdm
from reconstructions.utils import cameras as c

colors=['blue', 'orange']

# =============================================================================
# actors = sfb(cell, axon='green', neurite_radius=10)
# for actor in actors:
#     ccf_scene.add(actor)
# =============================================================================

for cell, color in zip(cells, colors):
    actors = sfb(cell, neurite_radius=10, axon=color)
    for actor in actors:
        ccf_scene.add(actor)
ccf_scene.render(camera=c.MYtopcam)
# =============================================================================
# for file in tqdm(os.listdir(celldir), desc='Loading cells'):
#     cell = file.split('.')[0]
#     if cell in cellLists.mossys:
#         fn = os.path.join(celldir, file)
#         actors = sfb(fn, neurite_radius=10)
#         for actor in actors:
#             ccf_scene.add(actor)
# =============================================================================

