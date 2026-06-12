from brainrender import Scene
from reconstructions.utils import load_data, preprocess_funcs, cameras
from reconstructions.utils.filedirs import allcoordswapped
from brainglobe_atlasapi import BrainGlobeAtlas
from brainrender.actors import Neuron, Points, ruler, Line
import brainrender
import numpy as np
import pandas as pd
import morphapi
from tqdm import tqdm
import os
import vedo
import json

# =============================================================================
# IRNdir = r'reconstructions\data\IRNPARN_cells\IRN'
# PARNdir = r'reconstructions\data\IRNPARN_cells\PARN'
# 
# _, IRNsomas, _, _, _ = load_data.load_neurons(IRNdir)
# _, PARNsomas, _, _, _ = load_data.load_neurons(PARNdir)
# 
# IRNsomas1 = np.array([value for _, value in IRNsomas.items()])
# PARNsomas1 = np.array([value for _, value in PARNsomas.items()])
# =============================================================================
somalist = []
for file in tqdm(os.listdir(allcoordswapped), desc='loading somas'):
    cellname = file.split('.')[0]
    filename = os.path.join(allcoordswapped, file)
    with open(filename, 'r') as f:
        neurondict = json.load(f)
        soma = neurondict['neurons'][0]['soma']
        somalist.append(soma)

somacoords = preprocess_funcs.get_coords(somalist, dim='all', mirror=True)
# =============================================================================
# 
# IRNsomacoords = preprocess_funcs.get_coords(IRNsomas, dim='all', mirror=True)
# PARNsomacoords = preprocess_funcs.get_coords(PARNsomas, dim='all', mirror=True)
# =============================================================================

ccf_scene = Scene(atlas_name='allen_mouse_10um', root=False)
ccf_scene.add_brain_region('IRN', color='red', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('PARN', color='blue', alpha=0.1, silhouette=False)
actors = ccf_scene.get_actors()
IRNmesh = actors[0]
IRNap = IRNmesh.mesh.vertices.T[0]
IRNdv = IRNmesh.mesh.vertices.T[1]
IRNml = IRNmesh.mesh.vertices.T[2]
IRNant = np.min(IRNap)
ANTirnbound = IRNant+800
linecoords = [[ANTirnbound, np.max(IRNdv), np.mean(IRNml)],[ANTirnbound, np.min(IRNdv), np.mean(IRNml)]]

#antplane = ccf_scene.atlas.get_plane(pos=(10000,4000,5000), plane='frontal')
medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
#ccf_scene.slice(plane=antplane)
ccf_scene.slice(plane=medplane)

somapoints = Points(somacoords, colors='red')

antRNline = Line(linecoords, color='black')
ccf_scene.add(somapoints)
ccf_scene.add(antRNline)
ccf_scene.render(camera=cameras.sagcam)