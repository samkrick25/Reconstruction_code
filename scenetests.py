# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:42:27 2026

@author: economolab
"""

from brainrender import Scene
from brainrender.actors import Neuron


cellpath=r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swc_noAA\swcN016-715345-HD.swc"
ccf_scene=Scene(atlas_name='allen_mouse_10um')
neuron=Neuron(neuron=cellpath)
ccf_scene.add(neuron)
ccf_scene.render()