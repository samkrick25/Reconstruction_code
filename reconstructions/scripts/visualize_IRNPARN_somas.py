from brainrender import Scene
from utils import load_data, preprocess_funcs, cameras
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

_, IRNsomas, _, _, _ = load_data.load_neurons(IRNdir)
_, PARNsomas, _, _, _ = load_data.load_neurons(PARNdir)

IRNsomas1 = np.array([value for _, value in IRNsomas.items()])
PARNsomas1 = np.array([value for _, value in PARNsomas.items()])

IRNsomacoords = preprocess_funcs.get_coords(IRNsomas1, dim='all')
PARNsomacoords = preprocess_funcs.get_coords(PARNsomas1, dim='all')

ccf_scene = Scene(atlas_name='allen_mouse_10um')
ccf_scene.add_brain_region('IRN', color='red', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('PARN', color='blue', alpha=0.1, silhouette=False)

antplane = ccf_scene.atlas.get_plane(pos=(10000,4000,5000), plane='frontal')
ccf_scene.slice(plane=antplane)

IRNpoints = Points(IRNsomacoords, colors='red')
PARNpoints = Points(PARNsomacoords, colors='blue')

ccf_scene.add(IRNpoints)
ccf_scene.add(PARNpoints)
ccf_scene.render(camera=cameras.corcam)