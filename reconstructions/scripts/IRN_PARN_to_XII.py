from brainrender import Scene
from reconstructions.utils import load_data as ld
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils import plotting_funcs as pl
from reconstructions.utils.filedirs import freqspkl, somaspkl, allcoordswapped
from reconstructions.utils.cameras import corcam, sagcam
from brainrender.actors import Points
import pickle
from tqdm import tqdm
import os
import numpy as np

freqs = pickle.load(open(freqspkl, 'rb'))
somas = pickle.load(open(somaspkl, 'rb'))
ccf_scene = Scene(atlas_name='allen_mouse_10um', root=True)
ccf_scene.add_brain_region('XII', color='blue', alpha=0.1, silhouette=False)
actors = ccf_scene.get_actors()
XIImesh = actors[1]
XIIvertices = XIImesh.mesh.vertices
XIIap = [vertex[0] for vertex in XIIvertices]
XIIdv = [vertex[1] for vertex in XIIvertices]
XIIlr = [vertex[2] for vertex in XIIvertices]
print(f'most posterior point of XII: {np.max(XIIap)}')
print(f'most anterior point of XII: {np.min(XIIap)}')
#input('Press Enter to continue')
XIIprojcells = [cell for cell, row in freqs.iterrows() if row['Ipsilateral XII'] + row['Contralateral XII'] > 3]
print(XIIprojcells)
XIIends={}
for file in tqdm(os.listdir(allcoordswapped), desc='Loading endpoints of XII+ cells'):
    cellname = file.split('.')[0]
    if cellname in XIIprojcells:
        endpoints = ld.get_endpoints_from_file(os.path.join(allcoordswapped, file))
        XIIends[cellname] = endpoints

#XIInodesbycell = pp.get_nodes_in_region(XIIends, 773, kind='by_cell')
#XIIcoords = ... #finish later, i can investigate this quesiton (topology of IRN/PARN->XII) by eye
XIInodes = pp.get_nodes_in_region(XIIends, 773, kind='bulk')

fig, axes = pl.plot_node_dist(XIInodes)
axes[0].axvline(x=13001.4, label='XII posterior boundary')
axes[0].axvline(x=12093.4, label='XII anterior boundary')
axes[1].axvline(x=np.max(XIIdv), label="XII ventral boundary")
axes[1].axvline(x=np.min(XIIdv), label="XII dorsal boundary")
axes[2].axvline(x=np.max(XIIlr), label="XII lateral boundary")
axes[2].axvline(x=5700, label="XII medial boundary (midline)")
fig.show()
input('Press Enter to continue')

#XIIcoords = pp.get_coords(XIInodes, dim='all')

#brainrender visualization stuff
# XIIpoints = Points(XIIcoords, colors='red')
#ccf_scene = Scene(atlas_name='allen_mouse_10um', root=True)
# actors = ccf_scene.get_actors() #just debug and look here to find actor indices
# root_ccf = actors[0]
# root_ccf._needs_silhouette = False
#ccf_scene.add_brain_region('XII', color='blue', alpha=0.1, silhouette=False)
#ccf_scene.add(XIIpoints)
#medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
#ccf_scene.slice(plane=medplane)
ccf_scene.render(camera=sagcam)