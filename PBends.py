# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:51:58 2026
plot ends in parabrachial nuc and color by phenotype
@author: samkr
"""
#%% imports
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.load_data import load_neurons, get_axonal_endpoints, get_nodes_in_region
from reconstructions.utils import filedirs
from reconstructions.utils.cellLists import cells_by_pheno

#%%
#set up paths and data structures
jsons = filedirs.allcoordswapped

ends_by_pheno = {}

pColors = {'sensory': 'firebrick', 'premotor': 'royalblue', 'mossy': 'seagreen', 'GRN': 'purple', 'sensorimotor': 'pink',
           'vestibular': 'navy', 'forebrain': 'orange'}

#%%
#create a mapping from cell name to pheno identity
def map_val_to_key(d):
    nd = {}
    for k, v in d.items():
        for val in v:
            nd[val]=k
    return nd

cells_to_pheno = map_val_to_key(cells_by_pheno)

#%% load cells
cells = load_neurons(jsons)

#%% get ends
ends_by_cell = get_axonal_endpoints(cells)
ends_by_cell = {cell: info['ends'] for cell, info in ends_by_cell.items()}

#%% get ends in PBN (or whtv your region of choice is)
PBends = get_nodes_in_region(ends_by_cell, regions='PB', ontlevel='structure', parcellated=False, mirror=True, kind='by_cell')

#%% get coordinates of each cell's ends to be put into brainrender
coords_by_cell = {cell: pp.get_coords(ends, dim='all') for cell, ends in PBends.items()}

#%% get point actors for each phenotype
import numpy as np
from brainrender.actors import Points

points_by_pheno = {'sensory': np.empty((0,3)),
                   'premotor': np.empty((0,3)),
                   'sensorimotor': np.empty((0,3)),
                   'GRN': np.empty((0,3)),
                   'forebrain': np.empty((0,3)),
                   'vestibular': np.empty((0,3)),
                   'mossy': np.empty((0,3)),
                   }
for cell, coords in coords_by_cell.items():
    if coords.size > 0:
        pheno = cells_to_pheno[cell]
        points_by_pheno[pheno] = np.append(points_by_pheno[pheno], coords, axis=0)

for pheno, points in points_by_pheno.items():
    if points.size == 0:    
        continue
    point_actors = Points(points, colors=pColors[pheno], radius=10)
    points_by_pheno[pheno] = point_actors

#%% set ccf scene, render points
from brainrender import settings, Scene
from reconstructions.utils import cameras as c

settings.INTERACTIVE = False
settings.OFFSCREEN = True
settings.BACKGROUND_COLOR = 'black'
settings.ROOT_ALPHA = 0.075
settings.ROOT_COLOR='black'


scene = Scene(atlas_name = 'allen_mouse_10um')

scene.get_actors()[0]._needs_silhouette = False

#add PB mesh, set silhouette
scene.add_brain_region('PB', color='black', alpha=0.01, silhouette=False)
PBmesh = scene.get_actors()[1]
scene.add_silhouette(PBmesh, color='white', lw=10)

#get 2 planes slightly offset in AP
PBc = PBmesh.center_of_mass()
PBv = PBmesh.vertices.T
PBx = PBv[0]
PBxmax = max(PBx)

aPos = [PBxmax-400, PBc[1], PBc[2]]
aPlane = scene.atlas.get_plane(pos=aPos, norm=[1,0,0])

pPos = [PBxmax-399, PBc[1], PBc[2]]
pPlane = scene.atlas.get_plane(pos=pPos, norm=[-1,0,0])

#slice scene w/planes to get 2d section
#scene.slice(aPlane)
#scene.slice(pPlane)

#add points
for pheno, points in points_by_pheno.items():
    if type(points) == np.ndarray:
        continue
    scene.add(points)

pbcam = dict(
    pos=(22982.6, 4105.88, -7143.51),
    focal_point=(5569.74, 4433.89, -7156.30),
    viewup=(0, -1.00000, 0),
    roll=180.000,
    distance=17416.0,
    clipping_range=(2934.33, 33358.4),
)

sf = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\images\PB\pheno_ends_in_PB_unsliced.png'
scene.screenshot(sf, camera=pbcam, scale=6)
scene.close()