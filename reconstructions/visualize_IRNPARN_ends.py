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

IRNdir = r'reconstructions\data\IRNPARN_cells\IRN'
PARNdir = r'reconstructions\data\IRNPARN_cells\PARN'

#621 is V, XII is 773, MRN 128
IRNends = load_data.load_endpoints(IRNdir)
PARNends = load_data.load_endpoints(PARNdir)

IRNallpoints = [point for cell in IRNends for point in cell]
PARNallpoints = [point for cell in PARNends for point in cell]

#add in other motor nuclei in a min, VII, VI, AMB, etc.
IRNmnends = preprocess_funcs.get_target_nodes_list(IRNallpoints, 128)
PARNmnends = preprocess_funcs.get_target_nodes_list(PARNallpoints, 128)

IRNcoords = preprocess_funcs.get_coords(IRNmnends, dim='all')
PARNcoords = preprocess_funcs.get_coords(PARNmnends, dim='all')

ccf_scene = Scene(atlas_name='allen_mouse_10um')
#ccf_scene.add_brain_region('XII', color='blue', alpha=0.1, silhouette=False)
#ccf_scene.add_brain_region('V', color='red', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('MRN', color='green', alpha=0.1, silhouette=False)

actors = ccf_scene.get_actors() #just debug and look here to find actor indices
root_ccf = actors[0]
root_ccf._needs_silhouette = False
allplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,0))
medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
ccf_scene.slice(plane=medplane)
ccf_scene.slice(plane=allplane, actors=actors[0])

IRNpoints = Points(IRNcoords, name='IRN endpoints', colors='red')
PARNpoints = Points(PARNcoords, name='PARN endpoints', colors='blue')

# medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
# ccf_scene.slice(plane=medplane)

#cameras
corcam = dict(
    pos=(85583.9, 3110.89, -5613.77),
    focal_point=(5572.12, 4618.09, -5672.54),
    viewup=(0, -1, 0),
    roll=179.966,
    distance=80026.0,
    clipping_range=(64035.5, 99934.5),
)
sagcam = dict(
    pos=(3404.93, 6.64762, 88515.2),
    focal_point=(5558.76, 4592.63, -7971.49),
    viewup=(0, -1, 0),
    roll=-179.774,
    distance=78000.0,
    clipping_range=(81177.1, 111749),
)

ccf_scene.add(IRNpoints)
ccf_scene.add(PARNpoints)
ccf_scene.render(camera=corcam)