# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:12:28 2026
rendering soma locations 2d coronal
@author: samkr
"""

from brainrender import Scene, settings
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils import cameras
from brainglobe_atlasapi import BrainGlobeAtlas as bga

# =============================================================================
# swcs =  [
#          r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\N067-685221-HS.swc",
#          r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\N068-685221.swc"
#          ]
# =============================================================================

swcs = [r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\N016-651324.swc"]


settings.SHOW_AXES = False
settings.OFFSCREEN=True
settings.INTERACTIVE=False

atlas = bga(atlas_name='allen_mouse_10um', check_latest=False)
my = atlas.get_structure_descendants('MY')
cb = atlas.get_structure_descendants('CB')
ft = atlas.get_structure_descendants('fiber tracts')
 
scene = Scene(atlas_name = 'allen_mouse_10um', check_latest=False)

scene.add_brain_region(*my, silhouette=True, alpha = 0.2, color='pink')
scene.add_brain_region(*cb, silhouette=True, alpha = 0.2, color='yellow')
scene.add_brain_region(*ft, silhouette=True, alpha=0.2, color='grey')

planeA = scene.atlas.get_plane(plane = 'frontal', pos = (11935.5572-1, 6415.2435, 	6683.219), norm= (1,0,0))
planeP = scene.atlas.get_plane(plane = 'frontal', pos = (11935.5572+1, 6415.2435, 	6683.219), norm = (-1,0,0))
scene.slice(planeA)
scene.slice(planeP)

scene.add_brain_region('IRN', silhouette=False, alpha=0.5, color='pink', force=True)

IRN = scene.get_actors()[-1]
scene.slice(planeA, actors=IRN, close_actors=True)
scene.slice(planeP, actors=IRN)
medplane=scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
scene.slice(plane=medplane, actors=IRN)

for swc in swcs:
    actors = pp.swap_for_brainrender(swc, skip_axon = True, skip_dendrite = True, soma = 'orange', soma_radius=30)
    for actor in actors:
        scene.add(actor)
    
    
frontcam = dict(
    pos=(-34007.2, -4240.85, -5572.80),
    focal_point=(12708.8, 4633.06, -5725.75),
    viewup=(0.186618, -0.982433, 2.17864e-4),
    roll=179.953,
    distance=47551.7,
    clipping_range=(45749.0, 49792.3),
)

savefile = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\images\cor_somas\N016-651324.png'
#scene.render(camera='frontal')
scene.screenshot(name=savefile, scale=3, camera=cameras.corcam)

#%%
from reconstructions.utils.filedirs import allcoordswapped, celldir
from brainrender import Scene, settings
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils import cameras
from brainglobe_atlasapi import BrainGlobeAtlas as bga
import os
import json
from tqdm import tqdm
import numpy as np

torender = ['N013-653980', 'N009-653980', 'N008-653980']

somas = {'x' : [], 'y' : [], 'z' : []}

for file in tqdm(os.listdir(allcoordswapped)):
    cellname = file.split('.')[0]
    if cellname in torender:
        ndict = json.load(open(os.path.join(allcoordswapped,file),'r'))
        sx, sy, sz = (ndict['neurons'][0]['soma']['x'], ndict['neurons'][0]['soma']['y'], ndict['neurons'][0]['soma']['z'])
        somas['x'].append(sx)
        somas['y'].append(sy)
        somas['z'].append(sz)

settings.SHOW_AXES = False
settings.OFFSCREEN=True
settings.INTERACTIVE=False

atlas = bga(atlas_name='allen_mouse_10um', check_latest=False)
my = atlas.get_structure_descendants('MY')
cb = atlas.get_structure_descendants('CB')
ft = atlas.get_structure_descendants('fiber tracts')
 
scene = Scene(atlas_name = 'allen_mouse_10um', check_latest=False)

scene.add_brain_region(*my, silhouette=True, alpha = 0.2, color='pink')
scene.add_brain_region(*cb, silhouette=True, alpha = 0.2, color='yellow')
scene.add_brain_region(*ft, silhouette=True, alpha=0.2, color='grey')

#do calculation to slice the scene here
mx = np.mean(somas['x'])
ay = np.mean(somas['y'])
az = np.mean(somas['z'])

planeA = scene.atlas.get_plane(plane = 'frontal', pos = (mx-.01, ay, az), norm= (1,0,0))
planeP = scene.atlas.get_plane(plane = 'frontal', pos = (mx+.01, ay, az), norm = (-1,0,0))
scene.slice(planeA)
scene.slice(planeP)

scene.add_brain_region('IRN', silhouette=False, alpha=0.5, color='pink', force=True)

IRN = scene.get_actors()[-1]
scene.slice(planeA, actors=IRN, close_actors=True)
scene.slice(planeP, actors=IRN)
medplane=scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
scene.slice(plane=medplane, actors=IRN)

for cell in tqdm(os.listdir(celldir),desc='Loading cells'):
    cellname = cell.split('.')[0]
    if cellname in torender:
        file = os.path.join(celldir, cell)
        actors = pp.swap_for_brainrender(file, skip_axon = True, skip_dendrite = True, soma = 'orange', soma_radius=30)
        for actor in actors:
            scene.add(actor)
            
frontcam = dict(
    pos=(-34007.2, -4240.85, -5572.80),
    focal_point=(12708.8, 4633.06, -5725.75),
    viewup=(0.186618, -0.982433, 2.17864e-4),
    roll=179.953,
    distance=47551.7,
    clipping_range=(45749.0, 49792.3),
)

savefile = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\images\cor_somas\653980somas.png'

scene.screenshot(savefile, scale=6, camera=frontcam)