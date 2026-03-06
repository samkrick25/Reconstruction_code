from brainrender import Scene
from reconstructions.utils import load_data as ld
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.filedirs import freqspkl, somaspkl, allcoordswapped
from reconstructions.utils.cameras import corcam, sagcam
from brainrender.actors import Points
import pickle
from tqdm import tqdm
import os

freqs = pickle.load(open(freqspkl, 'rb'))
somas = pickle.load(open(somaspkl, 'rb'))

XIIprojcells = [cell for cell, row in freqs.iterrows() if row['Ipsilateral XII'] + row['Contralateral XII'] > 3]
print(XIIprojcells)
XIIends={}
for file in tqdm(os.listdir(allcoordswapped), desc='Loading endpoints of XII+ cells'):
    cellname = file.split('.')[0]
    if cellname in XIIprojcells:
        endpoints = ld.get_endpoints_from_file(os.path.join(allcoordswapped, file))
        XIIends[cellname] = endpoints

XIInodes = pp.get_nodes_in_region(XIIends, 773)

XIIcoords = pp.get_coords(XIInodes, dim='all')

XIIpoints = Points(XIIcoords, colors='red')

ccf_scene = Scene(atlas_name='allen_mouse_10um', root=True)
actors = ccf_scene.get_actors() #just debug and look here to find actor indices
root_ccf = actors[0]
root_ccf._needs_silhouette = False
ccf_scene.add_brain_region('XII', color='blue', alpha=0.1, silhouette=False)
ccf_scene.add(XIIpoints)
# medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
# ccf_scene.slice(plane=medplane)
ccf_scene.render(camera=corcam)