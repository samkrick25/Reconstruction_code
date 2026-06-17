# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:42:33 2026

@author: economolab
"""

from brainrender import Scene
from reconstructions.utils import preprocess_funcs as pp

fn = r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\AA1521.swc"

ccf_scene = Scene(atlas_name='allen_mouse_10um')
ccf_scene.add_brain_region('MRN', alpha=0.2, color='purple', silhouette=False)

MRN = ccf_scene.atlas.get_region('MRN')
MRNcopy = MRN.mesh.clone()
MRNL = pp.get_mesh_onehem(ccf_scene, mesh=MRNcopy, hem='left')
MRNvertsL = MRNL.vertices


actors = pp.swap_for_brainrender(fn, axon='black', mesh=MRNvertsL, skip_dendrite=True)

for actor in actors:
    ccf_scene.add(actor)
    
ccf_scene.render()