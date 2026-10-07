# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:42:27 2026

@author: economolab
"""

from brainrender import Scene
from brainrender.actors import Neuron
from reconstructions.utils import visualize as vis


#cellpath=r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swc_noAA\swcN016-715345-HD.swc"
ccf_scene=Scene(atlas_name='allen_mouse_10um')

dvplane = ccf_scene.atlas.get_plane(pos=[0,4750,0], plane='horizontal')

ccf_scene.slice(dvplane)
#neuron=Neuron(neuron=cellpath)
#ccf_scene.add(neuron)
swc = 

ccf_scene.render()