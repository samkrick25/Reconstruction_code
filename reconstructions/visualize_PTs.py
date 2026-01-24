from brainrender import Scene
from utils import load_data
from brainglobe_atlasapi import BrainGlobeAtlas
from brainrender.actors import Neuron
import numpy as np
import pandas as pd
import morphapi
from tqdm import tqdm
import os
import vedo

diru1 = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\PTs\uppers\swc\upper1'
diru2 = r'C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\PTs\uppers\swc\upper2'
upper1s = []
upper2s = []

#load upper1s
for file in tqdm(os.listdir(diru1)):
    filename = os.path.join(diru1, file)
    neuron = Neuron(neuron=filename, color='blue')
    upper1s.append(neuron)

#load upper2s
for file in tqdm(os.listdir(diru2)):
    filename = os.path.join(diru2, file)
    neuron = Neuron(neuron=filename, color='red')
    upper2s.append(neuron)

#set scene and add meshes
ccf_scene = Scene(atlas_name='allen_mouse_10um')
ccf_scene.add_brain_region('STR', color='blue', alpha=0.05)
ccf_scene.add_brain_region('GPe', color='green', alpha=0.05)
for cell in upper1s:
    ccf_scene.add(cell)
for cell in upper2s:
    ccf_scene.add(cell)

antplane=ccf_scene.atlas.get_plane(pos=(3250,4000,5000),plane='frontal')
posplane=ccf_scene.atlas.get_plane(pos=(7750,4000,5000),norm=(-1, 0, 0),plane='frontal')
ccf_scene.slice(plane=antplane)
ccf_scene.slice(plane=posplane)


ccf_scene.render()
