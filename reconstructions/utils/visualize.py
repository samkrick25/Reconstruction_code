# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:23:38 2026
code to visualize IRN/PARN cells with brainrender
@author: economolab
"""

from brainrender import Scene
from reconstructions.utils import load_data as ld
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.filedirs import somaspkl, allcoordswapped, parcellated_neurons
import reconstructions.utils.cameras as cams
from brainrender.actors import Points
import pickle
from tqdm import tqdm
import os
import numpy as np

#get meshes for IRN/PARN from brainrender Scene
scene_for_bounds = Scene(atlas_name='allen_mouse_10um', root=False)
scene_for_bounds.add_brain_region('IRN')
scene_for_bounds.add_brain_region('PARN')
actors = scene_for_bounds.get_actors()
IRNmesh = actors[0]
IRNvertices = IRNmesh.mesh.vertices
PARNmesh = actors[1]
PARNvertices = PARNmesh.mesh.vertices

# =============================================================================
# #find anterior bound and set antIRN/PARN posterior bound
# IRNap = [vertex[0] for vertex in IRNvertices]
# PARNap = [vertex[0] for vertex in PARNvertices]
# IRNPARNap = IRNap+PARNap
# MedRNa_bound = np.min(IRNPARNap)
# MedRNp_bound = np.max(IRNPARNap)
# MedRNrange = MedRNp_bound-MedRNa_bound
# MedRNmid = MedRNa_bound + 800
# =============================================================================

#dorsal ventral separation
IRNdv = [vertex[1] for vertex in IRNvertices]
PARNdv = [vertex[1] for vertex in PARNvertices]
RNdv = IRNdv+PARNdv
dorsbound = np.min(RNdv)
ventbound = np.max(RNdv)
med = np.mean([int(dorsbound), int(ventbound)])

#get lists of antIRN/PARN vs postIRN/PARN cells
# =============================================================================
# somas = pickle.load(open(somaspkl, 'rb'))
# antIPARNcells = []
# postIPARNcells = []
# for cell, soma in somas.items():
#     if MedRNa_bound <= soma['x'] < MedRNmid:
#         antIPARNcells.append(cell)
#     if MedRNmid <= soma['x'] <= MedRNp_bound:
#         postIPARNcells.append(cell)
# =============================================================================
        

somas = pickle.load(open(somaspkl, 'rb'))
antIPARNcells = []
postIPARNcells = []
for cell, soma in somas.items():
    if dorsbound <= soma['x'] < med:
        antIPARNcells.append(cell)
    if med <= soma['x'] <= ventbound:
        postIPARNcells.append(cell)
        
        
#get endpoints of IRN/PARN cells split by their A->P position
antIPARNends = {}
postIPARNends = {}
for file in tqdm(os.listdir(parcellated_neurons), desc='Loading endpoints'):
    cellname = file.split('.')[0]
    endpoints = ld.get_endpoints_from_file_parcellated(os.path.join(parcellated_neurons, file))
    if cellname in antIPARNcells:
        antIPARNends[cellname] = endpoints
    if cellname in postIPARNcells:
        postIPARNends[cellname] = endpoints

def show_endpoints(*regions, root, colors, alphas, camera=cams.sagcam, unilat=False):
    '''
    visualize endpoints of IRN/PARN cells in any region defined by allen ccf 10um v3 
    (will rewrite to use other resolutions, binary expansion of regions and visualization of those)

    Parameters
    ----------
    *regions : str
        str of regions to be rendered and endpoints to be selected for visualization
    root : bool
        to be passed to brainrender.Scene.render, bool to render wholebrain mesh or not
    colors: list
        list of colors for each region to be rendered as, colors and regions must be the same length
    alphas: list
        list of alpha values for each region
    camera: reconstructions.utils.cameras variable
        what camera to use when rendering Scene, default is saggital view
    
    

    Returns
    -------
    None.

    '''

    if len(regions) != len(colors):
        raise ValueError('region and color list must be of same length!')
    if len(regions) != len(alphas):
        raise ValueError('region and alpha list must be of same length!')
        
    ccf_scene = Scene(atlas_name='allen_mouse_10um', root=root)
    for region, color, alpha in zip(regions, colors, alphas):
        ccf_scene.add_brain_region(region, color=color, alpha=alpha, silhouette=False)
    
    #this is messing up, returning empty lists
    antIPARNinreg = pp.get_nodes_in_region(antIPARNends, regions, parcellated=True, kind='by_cell', infunc=True)
    postIPARNinreg = pp.get_nodes_in_region(postIPARNends, regions, parcellated=True, kind='by_cell', infunc=True)
    
    antIPARNXIIswapped = pp.tenmicron_to_one(antIPARNinreg, allcoordswapped)
    postIPARNXIIswapped = pp.tenmicron_to_one(postIPARNinreg, allcoordswapped)
    #finish cleaning up from here later, need to rewrite this a little since above is now by cell, first need to look at coordswapped for actual coords
    antIPARNXIIcoords = []
    for cell, nodes in antIPARNXIIswapped.items():
        if unilat:
            coords = pp.get_coords(nodes, mirror=True)
        else:
            coords = pp.get_coords(nodes, mirror=False)
        if coords.any():
            antIPARNXIIcoords.append(coords)
    antIPARNXIIflat = np.array([node for cell in antIPARNXIIcoords for node in cell])

    postIPARNXIIcoords = []
    for cell, nodes in postIPARNXIIswapped.items():
        if unilat:
            coords = pp.get_coords(nodes, mirror=True)
        else:
            coords = pp.get_coords(nodes, mirror=False)
        if coords.any():
            postIPARNXIIcoords.append(coords)
    postIPARNXIIflat = np.array([node for cell in postIPARNXIIcoords for node in cell])
    
    antPoints = Points(antIPARNXIIflat, colors='red', radius=10)
    postPoints = Points(postIPARNXIIflat, colors='blue', radius=10)
    ccf_scene.add(antPoints)
    ccf_scene.add(postPoints)
    
    if unilat:
        medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
        ccf_scene.slice(plane=medplane)
    
    ccf_scene.render(camera=camera)
    
    return antIPARNXIIswapped, postIPARNXIIswapped