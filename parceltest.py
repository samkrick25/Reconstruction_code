# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:12:24 2026
testing parcellation stuff
@author: samkr
"""

from reconstructions.utils import preprocess_funcs as pp
from brainrender.actors import Points
from brainrender import Scene
import json

nj = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\json_coordswapped\N039-674185-IB.json"

cs = Scene(atlas_name='allen_mouse_10um')
root = cs.get_actors()[0]
root._needs_silhouette=False
cs.add_brain_region('P5', silhouette=False, alpha=0.2)
cs.add_brain_region('V', silhouette=False, alpha=0.1)

with open(nj, 'r') as f:
    neuron = json.load(f)
    axon = neuron['neurons'][0]['axon']

tspn = pp.get_target_nodes_list(axon, 549009215)

tspnc = pp.get_coords(tspn, dim='all')

tspp = Points(tspnc, colors='black', radius=3)

cs.add(tspp)
cs.render()
    