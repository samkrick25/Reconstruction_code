# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:18:17 2026
THIS FILE DOENS'T WORK, might fix at a later date idk
@author: economolab
"""

from brainrender import Scene
from brainrender import settings
import pandas as pd
import numpy as np
import os
from itertools import cycle, islice
from tqdm import tqdm
from reconstructions.utils import cameras
from pathlib import Path
from vedo import Lines
import pickle
from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import preprocess_funcs as pf

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
celldir = r"reconstructions\data\IRNPARN_cells\swcsfromjson"#C:\Users\economolab\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson"
savedir = r"images\all_cells\withtoptargets"#C:\Users\economolab\Documents\GitHub\Reconstruction_code\images\all_cells"

#set colors for each cell to be rendered as
cellcolors = ['blue','red','orange','green','purple']
output = list(islice(cycle(cellcolors), len(os.listdir(celldir)))) 

#load frequency data to decide which regions to render
frequencies_notprocessed = pickle.load(open(frequenciespkl, 'rb'))
fnp = frequencies_notprocessed.T

#turn off brainrender axes
settings.SHOW_AXES=None

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
    ccf_scenesag = Scene(atlas_name='allen_mouse_10um', root=False)
    ccf_scenesag.add_brain_region('IRN', color='pink', silhouette=False, alpha=0)
    ccf_scenesag.add_brain_region('PARN', color='pink', silhouette=False, alpha=0)
    

    
    #set directories/filepaths
    filestr = file.split('.')[0]
    cellname = filestr#[3:]
    filename = os.path.join(celldir, file)
    cellsavedir = os.path.join(savedir, cellname)
    Path(cellsavedir).mkdir(parents=True, exist_ok=True)
    legendsavefile = os.path.join(cellsavedir, cellname+'_legend.png')
    
    #find top 5 targeted regions (excluding IRN/PARN and any unassigned) and render them
    regcolors = ['orange', 'purple', 'cyan', 'green', 'red']
    targets = []
    targseries = pf.get_targeted_regions(fnp, cellname)
    for region in targseries.index:
        abv = region.split(' ')[-1]
        if 'IRN' in abv:
            continue
        if 'PARN' in abv:
            continue
        if 'unassigned' in abv:
            continue
        else:
            targets.append(abv)
    top5 = targets[0:5]
    lastactcoords = []
    for reg, color in zip(top5, regcolors):
        ccf_scenesag.add_brain_region(reg, color=color, silhouette=False, alpha=0)
        regactor = ccf_scenesag.get_actors()[-1]
        
        #get meshes highest point(where the label will be added)
# =============================================================================
#         default_offset = np.array([0, -200, 100])
#         points = regactor.mesh.vertices.copy()
#         point = points[np.argmin(points[:, 1]), :]
#         point += default_offset
#         point[2] = -point[2]
#         xpos, ypos, zpos = point
# =============================================================================
        if lastactcoords:
            lastactcoords[0] -= 500
            #DEBUG THIS THE NEWPOINT IS BROKEN
            ccf_scenesag.add_label(regactor, reg, color=color, radius=0, yrot=90, newpoint=lastactcoords)
        else:
            ccf_scenesag.add_label(regactor, reg, color=color, radius=0, yrot=90)
        labelact = ccf_scenesag.get_actors()[-1]
        labelap = [v[0] for v in labelact.mesh.vertices]
        labeldv = [v[1] for v in labelact.mesh.vertices]
        labelml = [v[2] for v in labelact.mesh.vertices]
            
        if len(lastactcoords) == 3:   
            lastactcoords=[]

            lastactcoords.append(np.min(labelap))
            lastactcoords.append(np.min(labeldv))
            lastactcoords.append(np.min(labelml))
        else:
            lastactcoords.append(np.min(labelap))
            lastactcoords.append(np.min(labeldv))
            lastactcoords.append(np.min(labelml))
    ccf_scenesag.render(camera=cameras.corcam)
# =============================================================================
#     ccf_scenesag.screenshot(name=legendsavefile, camera=cameras.corcam) #(camera=cameras.corcam)
#     ccf_scenesag.close()
# 
# =============================================================================
