from brainrender import Scene
from utils import load_data, preprocess_funcs
from brainglobe_atlasapi import BrainGlobeAtlas
from brainrender.actors import Neuron, Points, ruler
import brainrender
import numpy as np
import pandas as pd
import morphapi
from tqdm import tqdm
import os
import vedo

#shared pc dirs below
#diru1 = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\PTs\uppers\json\upper1'
#diru2 = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\PTs\uppers\json\upper2'
#my laptop dirs
diru1 = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\PTs\uppers\json\upper1'
diru2 = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\PTs\uppers\json\upper2'

upper1s = load_data.load_endpoints(diru1)
upper2s = load_data.load_endpoints(diru2) 

#load coordinates of cells in target ROI
#1022 gpe, 672 str
#931 pontine gray
#PF 930, MD 362, CL 575, LP 218, PO 1020
u1targetnodes = [preprocess_funcs.get_target_nodes_list(cell, 930, 362, 575, 218, 1020) for cell in upper1s]
u1coords = []
for cell in u1targetnodes:
    if cell:
        coords = preprocess_funcs.get_coords(cell, dim='all')  
        u1coords.append(coords)
    else:
        continue
u1coords = [point for cell in u1coords for point in cell]

u2targetnodes = [preprocess_funcs.get_target_nodes_list(cell, 930, 362, 575, 218, 1020) for cell in upper2s]
u2coords = []
for cell in u2targetnodes:
    if cell:
        u2coords.append(preprocess_funcs.get_coords(cell, dim='all'))
    else:
        continue
u2coords = [point for cell in u2coords for point in cell]

#add in a ruler for scale bar maybe? ill have to adjust and pick random points until i find something that works
p1 = [3000, 5000, 4000]
p2 = [2000, 5000, 4000]
ruler = ruler(np.array(p1, dtype=np.int64), np.array(p2, dtype=np.int64), units='um', s=0)

#set scene and add region meshes
ccf_scene = Scene(atlas_name='allen_mouse_10um') #, title='PT upper endpoint distribution in GPe and STR'
# ccf_scene.add_brain_region('STR', color='red', alpha=0.05, silhouette=False)
# ccf_scene.add_brain_region('GPe', color='blue', alpha=0.05, silhouette=False)
#ccf_scene.add_brain_region('PG', color='orange', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('CL', color='red', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('PF', color='blue', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('MD', color='cyan', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('LP', color='yellow', alpha=0.2, silhouette=False)
ccf_scene.add_brain_region('PO', color='orange', alpha=0.1, silhouette=False)

#actor[2] is STR, actor[3] is GPe, actor[4] is upper 1, 5 is upper 2
actors = ccf_scene.get_actors() #just debug and look here to find actor indices
root_ccf = actors[0]
root_ccf._needs_silhouette = False #turn off silhouette for the wholebrain mesh
#root_ccf._is_added = True
#root_ccf._alpha
#print(actors[0])
#actor labels here for BG visualization
#ccf_scene.add_label(actors[1], label='STR', xoffset=0, zoffset=1000, yoffset=4000, radius=None, color='light blue')
#ccf_scene.add_label(actors[2], label='GPe', xoffset=0, zoffset=2000, yoffset=3950, radius=None, color='light green')
#ccf_scene.add_label(actors[2], label='PT upper 1', radius=None, color='blue', xoffset=-12000, zoffset=250, yoffset=7880) #for bg, xoffset=0, zoffset=500, yoffset=3560
#ccf_scene.add_label(actors[3], label='PT upper 2', radius=None, color='red', xoffset=-12000, zoffset=-200, yoffset=8055) #for bg, xoffset=0, zoffset=-250, yoffset=3050
#ccf_scene.add_label(actors[1], label='PG', radius=None, color='orange', xoffset=-12000, zoffset=500, yoffset=8000)

#slice scene to show ROI
#antplanebg=ccf_scene.atlas.get_plane(pos=(3250,4000,5000),plane='frontal')
#posplanebg=ccf_scene.atlas.get_plane(pos=(7750,4000,5000),norm=(-1, 0, 0),plane='frontal')
#antplanepg=ccf_scene.atlas.get_plane(pos=(8500,4000,5000),plane='frontal')
#posplanepg=ccf_scene.atlas.get_plane(pos=(9500,4000,5000),norm=(-1,0,0),plane='frontal')
#medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
allplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,0))
#ccf_scene.slice(plane=antplanebg)
#ccf_scene.slice(plane=posplanebg)
#ccf_scene.slice(plane=medplane)
ccf_scene.slice(plane=allplane, actors=actors[0])

#load and add points
u1points = Points(np.array(u1coords), name='PT upper 1', colors='blue')
u2points = Points(np.array(u2coords), name='PT upper 2', colors='red')
ccf_scene.add(u1points)
ccf_scene.add(u2points)
#ccf_scene.add(ruler)

#set custom camera parameters
sagcam = dict(
    pos=(3404.93, 6.64762, 88515.2),
    focal_point=(5558.76, 4592.63, -7971.49),
    viewup=(0, -1, 0),
    roll=-179.774,
    distance=78000.0,
    clipping_range=(81177.1, 111749),
)
corcam = dict(
    pos=(85583.9, 3110.89, -5613.77),
    focal_point=(5572.12, 4618.09, -5672.54),
    viewup=(0, -1, 0),
    roll=179.966,
    distance=80026.0,
    clipping_range=(64035.5, 99934.5),
)
pgcam = dict(
    pos=(87053.0, -38138.4, -5822.37),
    focal_point=(6496.29, 6061.31, -6020.54),
    viewup=(-0.481008, -0.876605, 0.0139624),
    roll=-180,
    distance=91886.0,
    clipping_range=(73566.7, 113064),
)
topcam = dict(
    pos=(7760.00, -31645.0, -5943.00),
    focal_point=(7829.74, 4296.03, -5694.50),
    viewup=(-1.00000, 0, 0),
    roll=74.3241,
    distance=35942.0,
    clipping_range=(26252.0, 46156.4),
)


ccf_scene.render(camera=corcam)
#ccf_scene.screenshot(name='PTs_in_STR_and_GPe')5