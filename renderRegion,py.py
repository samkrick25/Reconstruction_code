# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:33:43 2026

@author: samkr
"""

from brainrender import Scene
from reconstructions.utils.preprocess_funcs import swap_for_brainrender as sfb
from reconstructions.utils import cellLists

celldir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson'

ccf_scene = Scene(atlas_name="allen_mouse_10um")
root = ccf_scene.get_actors()[0]
root._needs_silhouette=False

regtoadd = ['MPO', 'ADP', 'VMH', 'LPO', 'NLOT', 'AVPV', 'DMH', 'PMv', 'ARH',
       'PS', 'LSr', 'AP', 'SBPV', 'MS', 'PMd', 'MPN', 'PVpo', 'COAa', 'PVH'
]
for reg in regtoadd:
    ccf_scene.add_brain_region(reg, silhouette=False, color='pink', alpha=0.25)

cells = [
         r"reconstructions\data\IRNPARN_cells\swcsfromjson\N040-709222-MB.swc",
         r"reconstructions\data\IRNPARN_cells\swcsfromjson\N016-651324.swc"
         ]

import os
from tqdm import tqdm
from reconstructions.utils import cameras as c

colors=['green', 'blue']

for cell, color in zip(cells, colors):
    actors = sfb(cell, neurite_radius=10, axon=color)
    for actor in actors:
        ccf_scene.add(actor)
ccf_scene.render(camera=c.topcam)
# =============================================================================
# for file in tqdm(os.listdir(celldir), desc='Loading cells'):
#     cell = file.split('.')[0]
#     if cell in cellLists.mossys:
#         fn = os.path.join(celldir, file)
#         actors = sfb(fn, neurite_radius=10)
#         for actor in actors:
#             ccf_scene.add(actor)
# =============================================================================

