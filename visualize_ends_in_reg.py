'''
render endpoints in any given region
need to find a way to make this faster
right now i have to index ccf for every endpoint and check if it is in desired region
i guess i could rewrite the jsons to be accurate but idk
'''

from brainrender import Scene, settings
from reconstructions.utils import load_data as ld
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.filedirs import allcoordswapped, allen_ccf_10um
from reconstructions.utils import cameras
from brainrender.actors import Points
import os
from tqdm import tqdm
import nibabel as nib
import numpy as np

allen_ccf = nib.load(allen_ccf_10um)
allen_ccf_data = np.asanyarray(allen_ccf.dataobj)

def get_nodes_in_region(endsdict, pidx):
    '''
    find nodes in given allen ccf regions

    Parameters
    ----------
    endsdict : dict
        Dictionary of nodes which is the output of load_data.get_endpoints_from_file().
    pidx : list
        allen parcellation indices of the regions you wish to pull nodes for.

    Returns
    -------
    list. nodes in region

    '''
    endsinreg = []
    for cell, info in tqdm(endsdict.items(), desc='Finding ends in reg'):
        ends = info['ends']
        for end in ends:
            coords = [end['x'], end['y'], end['z']]
            tenmicron = np.round([x/10 for x in coords]).astype(int).tolist()
            try:
 
                allenid = allen_ccf_data[tenmicron[0], tenmicron[1], tenmicron[2]]
            #nodes that are in the spinal cord will throw an error since the spinal cord doesn't exist in allen CCF
            except IndexError:
                continue
            if allenid in pidx:
                endsinreg.append(end)
               
    return endsinreg

cellends = {}
for file in tqdm(os.listdir(allcoordswapped), desc='Loading endpoints'):
    cellname = file.split('.')[0]
    endpoints = ld.get_endpoints_from_file(os.path.join(allcoordswapped, file))
    cellends[cellname] = endpoints[cellname]

# GRNends = [ld.get_endpoints_from_file(file) for file in os.listdir(allcoordswapped) if file.split(r'\\')[-1] in GRNpos]
#print(GRNends)
# %%

print('getting nodes in reg')
#the second param should be a list of allen parcellation indices of the regions 
#you want to find endpoints in
endsinreg = get_nodes_in_region(cellends, [857])

print('getting coords')
coords = pp.get_coords(endsinreg, dim='all', mirror=True)



points = Points(coords, colors='green')

settings.ROOT_ALPHA = 0

ccf_scene = Scene(atlas_name='allen_mouse_10um', root=False)
#ccf_scene.add_brain_region('MRN', color='pink', alpha=0.1, silhouette=False)
ccf_scene.add_brain_region('PB', color='orange', alpha=0.1, silhouette=False)
ccf_scene.add(points)
medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
ccf_scene.slice(plane=medplane)
ccf_scene.render(camera=cameras.corcam)
