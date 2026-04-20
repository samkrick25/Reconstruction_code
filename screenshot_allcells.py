# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:18:17 2026
unsure why, but can only take either top or sag screenshots, cant do both of them at once
@author: economolab
"""

from brainrender import Scene
import pandas as pd
import os
from itertools import cycle, islice
from tqdm import tqdm
from reconstructions.utils import cameras
from pathlib import Path
from vedo import Lines

#claude help, generating line actors for axon and dendrite instead of using morphapi tube mesh generation since that is deleting some nodes
def swc_to_line_actors(swc_df, axon_color='blue', dendrite_color='red', lw=2):
    """
    Build vedo Lines actors for axon and dendrite segments directly
    from a parsed SWC dataframe. Bypasses morphapi/vedo tube merging.
    """
    node_coords = swc_df.set_index('id')[['x', 'y', 'z']]

    # Only rows that have a valid parent
    has_parent = swc_df[swc_df['parent'] != -1]

    actors = []
    for neurite_type, color in [(2, axon_color), (3, dendrite_color)]:
        subset = has_parent[has_parent['type'] == neurite_type]
        if subset.empty:
            continue

        # Build (N, 3) start and end point arrays
        start_pts = node_coords.loc[subset['id']].values
        end_pts   = node_coords.loc[subset['parent']].values

        actor = Lines(start_pts, end_pts, c=color, lw=lw)
        actors.append(actor)

    return actors

#set directories
celldir = r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson"
savedir = r"C:\Users\economolab\Documents\GitHub\Reconstruction_code\images\all_cells"

#set colors for each cell to be rendered as
cellcolors = ['blue','red','orange','green','purple']
output = list(islice(cycle(cellcolors), len(os.listdir(celldir))))


#brainrender scene for top screenshots
# =============================================================================
# ccf_scenetop = Scene(atlas_name='allen_mouse_10um')
# ccf_scenetop.add_brain_region('IRN', color='pink', silhouette=False, alpha=0.2)
# ccf_scenetop.add_brain_region('PARN', color='pink', silhouette=False, alpha=0.2)
# medplane=ccf_scenetop.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
# ccf_scenetop.slice(plane=medplane)
# topact = ccf_scenetop.get_actors()
# roottop = topact[0]
# roottop._needs_silhouette=False
# =============================================================================

#iterate over and screenshot neurons, save each to their own folder
for file, color in tqdm(zip(os.listdir(celldir),output),desc='loading and screenshotting cells'):
    #set up brainrender scene for sag screenshots
    ccf_scenesag = Scene(atlas_name='allen_mouse_10um')
    ccf_scenesag.add_brain_region('IRN', color='pink', silhouette=False, alpha=0.2)
    ccf_scenesag.add_brain_region('PARN', color='pink', silhouette=False, alpha=0.2)
    medplane=ccf_scenesag.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
    #ccf_scenesag.slice(plane=medplane)
    actors = ccf_scenesag.get_actors()
    rootsag=actors[0]
    rootsag._needs_silhouette = False
    
    #set directories/filepaths
    filestr = file.split('.')[0]
    cellname = filestr#[3:]
    filename = os.path.join(celldir, file)
    cellsavedir = os.path.join(savedir, cellname)
    Path(cellsavedir).mkdir(parents=True, exist_ok=True)
    sagsavefile = os.path.join(cellsavedir, cellname+'_sag.png')
    topsavefile = os.path.join(cellsavedir, cellname+'_top.png')
    
    #add neuron, screenshot from saggital view
# =============================================================================
#     neuronsag = Neuron(neuron=filename, color=color)
#     ccf_scenesag.add(neuronsag)
# =============================================================================
    swc_df = pd.read_csv(
        filename,
        comment='#',
        sep=r'\s+',
        names=['id', 'type', 'x', 'y', 'z', 'r', 'parent']
    )
    x = swc_df['x']
    z = swc_df['z']
    swc_df['x'] = z
    swc_df['z'] = x
    cell_actors = swc_to_line_actors(swc_df, axon_color=color, dendrite_color='black', lw=2)
    #screenshot (change name/camera for different views)
    for actor in cell_actors:
        ccf_scenesag.add(actor)
    ccf_scenesag.screenshot(name=topsavefile, camera=cameras.MYtopcam)
# =============================================================================
#     for actor in cell_actors:
#         ccf_scenesag.remove(actor)
# =============================================================================


