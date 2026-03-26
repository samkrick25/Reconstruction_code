from brainrender import Scene
from reconstructions.utils import load_data as ld
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils import plotting_funcs as pl
from reconstructions.utils.filedirs import freqspkl, somaspkl, allcoordswapped, parcellated_neurons
import reconstructions.utils.cameras as cams
from brainrender.actors import Points
import pickle
from tqdm import tqdm
import os
import numpy as np


#os.system('conda activate reconstructions')

freqs = pickle.load(open(freqspkl, 'rb'))
somas = pickle.load(open(somaspkl, 'rb'))

ccf_scene = Scene(atlas_name='allen_mouse_10um', root=False)
ccf_scene.add_brain_region('XII', color='orange', alpha=0.2, silhouette=False)
ccf_scene.add_brain_region('IRN', color='red', alpha=0.0, silhouette=False)
ccf_scene.add_brain_region('PARN', color='pink', alpha=0.0, silhouette=False)

actors = ccf_scene.get_actors()
XIImesh = actors[0]
IRNmesh = actors[1]
PARNmesh = actors[2]
XIIvertices = XIImesh.mesh.vertices
IRNvertices = IRNmesh.mesh.vertices
PARNvertices = PARNmesh.mesh.vertices

IRNap = [vertex[0] for vertex in IRNvertices]
PARNap = [vertex[0] for vertex in IRNvertices]
IRNPARNap = IRNap+PARNap
MedRNa_bound = np.min(IRNPARNap)
MedRNp_bound = np.max(IRNPARNap)
MedRNrange = MedRNp_bound-MedRNa_bound
MedRNmid = MedRNa_bound + 800
#MedRNmid2 = MedRNa_bound + 2*(MedRNmid1)

antIPARNcells = []
postIPARNcells = []
for cell, soma in somas.items():
    if MedRNa_bound <= soma['x'] < MedRNmid:
        antIPARNcells.append(cell)
    if MedRNmid <= soma['x'] <= MedRNp_bound:
        postIPARNcells.append(cell)
        
antIPARNends = {}
postIPARNends = {}
for file in tqdm(os.listdir(parcellated_neurons), desc='Loading endpoints'):
    cellname = file.split('.')[0]
    endpoints = ld.get_endpoints_from_file_parcellated(os.path.join(parcellated_neurons, file))
    if cellname in antIPARNcells:
        antIPARNends[cellname] = endpoints
    if cellname in postIPARNcells:
        postIPARNends[cellname] = endpoints
        
antIPARNXII = pp.get_nodes_in_region(antIPARNends, 'XII', parcellated=True, kind='by_cell')
postIPARNXII = pp.get_nodes_in_region(postIPARNends, 'XII', parcellated=True, kind='by_cell')

antIPARNXIIswapped = pp.tenmicron_to_one(antIPARNXII, allcoordswapped)
postIPARNXIIswapped = pp.tenmicron_to_one(postIPARNXII, allcoordswapped)
#finish cleaning up from here later, need to rewrite this a little since above is now by cell, first need to look at coordswapped for actual coords
antIPARNXIIcoords = []
for cell, nodes in antIPARNXIIswapped.items():
    coords = pp.get_coords(nodes, mirror=True)
    if coords.any():
        antIPARNXIIcoords.append(coords)
antIPARNXIIflat = np.array([node for cell in antIPARNXIIcoords for node in cell])

postIPARNXIIcoords = []
for cell, nodes in postIPARNXIIswapped.items():
    coords = pp.get_coords(nodes, mirror=True)
    if coords.any():
        postIPARNXIIcoords.append(coords)
postIPARNXIIflat = np.array([node for cell in postIPARNXIIcoords for node in cell])

XIIap = [vertex[0] for vertex in XIIvertices]
XIIdv = [vertex[1] for vertex in XIIvertices]
XIIlr = [vertex[2] for vertex in XIIvertices]
#input('Press Enter to continue')
# =============================================================================
# XIIprojcells = [cell for cell, row in freqs.iterrows() if row['Ipsilateral XII'] + row['Contralateral XII'] > 3]
# print(XIIprojcells)
# XIIends={}
# for file in tqdm(os.listdir(allcoordswapped), desc='Loading endpoints of XII+ cells'):
#     cellname = file.split('.')[0]
#     if cellname in XIIprojcells:
#         endpoints = ld.get_endpoints_from_file(os.path.join(allcoordswapped, file))
#         XIIends[cellname] = endpoints
# =============================================================================

#XIInodesbycell = pp.get_nodes_in_region(XIIends, 773, kind='by_cell')
#XIIcoords = ... #finish later, i can investigate this quesiton (topology of IRN/PARN->XII) by eye
#XIInodes = pp.get_nodes_in_region(XIIends, 773, kind='bulk')


#XIIcoords = pp.get_coords(XIInodes, dim='all')

#brainrender visualization stuff
#XIIpoints = Points(XIIcoords, colors='red')
antPoints = Points(antIPARNXIIflat, colors='red', radius=10)
postPoints = Points(postIPARNXIIflat, colors='blue', radius=10)
# =============================================================================
# actors = ccf_scene.get_actors() #just debug and look here to find actor indices
# root_ccf = actors[0]
# root_ccf._needs_silhouette = False
# =============================================================================
#ccf_scene.add_brain_region('XII', color='blue', alpha=0.1, silhouette=False)
#ccf_scene.add(XIIpoints)
ccf_scene.add(antPoints)
ccf_scene.add(postPoints)
medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
ccf_scene.slice(plane=medplane)
ccf_scene.render(camera=cams.sagcam)