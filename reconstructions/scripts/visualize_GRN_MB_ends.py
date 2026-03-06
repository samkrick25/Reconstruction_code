from brainrender import Scene
from reconstructions.utils import load_data as ld
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.filedirs import allcoordswapped, GRNpos
from reconstructions.utils.cameras import sagcam
from brainrender.actors import Points
import os
from tqdm import tqdm

GRNends = {}
for file in tqdm(os.listdir(allcoordswapped), desc='Loading endpoints of GRN+ cells'):
    cellname = file.split('.')[0]
    if cellname in GRNpos:
        endpoints = ld.get_endpoints_from_file(os.path.join(allcoordswapped, file))
        GRNends[cellname] = endpoints

# GRNends = [ld.get_endpoints_from_file(file) for file in os.listdir(allcoordswapped) if file.split(r'\\')[-1] in GRNpos]
#print(GRNends)

GRNendsin_MRN = pp.get_nodes_in_region(GRNends, 128)

GRNcoordsinMRN = pp.get_coords(GRNendsin_MRN, dim='all')

GRNpointsinMRN = Points(GRNcoordsinMRN, colors='green')

ccf_scene = Scene(atlas_name='allen_mouse_10um', root=False)
ccf_scene.add_brain_region('MRN', color='pink', alpha=0.1, silhouette=False)
ccf_scene.add(GRNpointsinMRN)
medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
ccf_scene.slice(plane=medplane)
ccf_scene.render(camera=sagcam)
