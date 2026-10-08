# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:42:27 2026

@author: economolab
"""

from brainrender import Scene, settings
from brainrender.actors import Neuron, Points
from reconstructions.utils import visualize as vis
import numpy as np

settings.SHOW_AXES = True

#cellpath=r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swc_noAA\swcN016-715345-HD.swc"
ccf_scene=Scene(atlas_name='allen_mouse_10um')

# =============================================================================
# ccf_scene.add_brain_region('CB')
# CBverts = ccf_scene.get_actors()[-1].mesh.vertices
# =============================================================================
rootverts = ccf_scene.get_actors()[0].mesh.vertices


dvplane = ccf_scene.atlas.get_plane(pos=[0,4640,0], plane='horizontal')

swc = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\N132-708369-SA.swc"

# =============================================================================
# actors = vis.create_cell_actors(swc)
# for actor in actors:
#     ccf_scene.add(actor)
# =============================================================================

point = Points(np.array([[13165.083252, 5787.026584, 4761.346117]]), radius=10)
ccf_scene.add(point)

#ccf_scene.add(dvplane)
#ccf_scene.slice(dvplane)
#neuron=Neuron(neuron=cellpath)
#ccf_scene.add(neuron)
ccf_scene.render()