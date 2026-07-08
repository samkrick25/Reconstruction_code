# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:18:17 2026
unsure why, but can only take either top or sag screenshots, cant do both of them at once
@author: economolab
"""

from brainrender import Scene
from brainrender import settings
import pandas as pd
import os
from itertools import cycle, islice
from tqdm import tqdm
from reconstructions.utils import cameras
from pathlib import Path
from vedo import Lines
import pickle
from reconstructions.utils.filedirs import frequenciespkl
from reconstructions.utils import preprocess_funcs as pf
from reconstructions.utils import cellLists

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
savedir = r"images\all_cells"#C:\Users\economolab\Documents\GitHub\Reconstruction_code\images\all_cells"


#set colors for each cell to be rendered as
cellcolors = ['blue','red','orange','green','purple']
output = list(islice(cycle(cellcolors), len(os.listdir(celldir)))) 

#load frequency data to decide which regions to render
frequencies_notprocessed = pickle.load(open(frequenciespkl, 'rb'))
fnp = frequencies_notprocessed.T

#turn off brainrender axes
settings.SHOW_AXES=None
settings.OFFSCREEN = True

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
    #set directories/filepaths
    filestr = file.split('.')[0]
    cellname = filestr#[3:]
    if cellname in cellLists.new709222:
        #set up brainrender scene for sag screenshots
        ccf_scenesag = Scene(atlas_name='allen_mouse_10um')
        ccf_scenesag.add_brain_region('IRN', color='pink', silhouette=False, alpha=0.15)
        ccf_scenesag.add_brain_region('PARN', color='pink', silhouette=False, alpha=0.15)
    # =============================================================================
    #     ccf_scenesag.add_brain_region('XII', color='green', silhouette=False, alpha=0.2)
    #     ccf_scenesag.add_brain_region('VII', color='cyan', silhouette=False, alpha=0.2)
    #     ccf_scenesag.add_brain_region('V', color='orange', silhouette=False, alpha=0.2)
    # =============================================================================
# =============================================================================
#         medplane=ccf_scenesag.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
#         ccf_scenesag.slice(plane=medplane)
# =============================================================================
        actors = ccf_scenesag.get_actors()
        rootsag=actors[0]
        rootsag._needs_silhouette = False


        filename = os.path.join(celldir, file)
        #if cellname in cellLists.heavy_premotors:
        cellsavedir = os.path.join(savedir, cellname)
        Path(cellsavedir).mkdir(parents=True, exist_ok=True)
        sagsavefile = os.path.join(cellsavedir, cellname+'_sag.png')
        topsavefile = os.path.join(cellsavedir, cellname+'_top.png')
        corsavefile = os.path.join(cellsavedir, cellname+'_cor.png')
            
        # =============================================================================
        #     #find top 5 targeted regions (excluding IRN/PARN and any unassigned) and render them (doesn't look very good)
        #     regcolors = ['orange', 'purple', 'cyan', 'green', 'red']
        #     targets = []
        #     targseries = pf.get_targeted_regions(fnp, cellname)
        #     for region in targseries.index:
        #         abv = region.split(' ')[-1]
        #         if 'IRN' in abv:
        #             continue
        #         if 'PARN' in abv:
        #             continue
        #         if 'unassigned' in abv:
        #             continue
        #         else:
        #             targets.append(abv)
        #     top5 = targets[0:5]
        #     for reg, color in zip(top5, regcolors):
        #         ccf_scenesag.add_brain_region(reg, color=color, silhouette=False, alpha=0.2)
        #         regactor = ccf_scenesag.get_actors()[-1]
        #         #ccf_scenesag.add_label(regactor, reg, color=color, radius=0, yrot=90)
        # =============================================================================
        #ccf_scenesag.slice(plane=medplane) 
        #add neuron, screenshot from saggital view
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
        ccf_scenesag.screenshot(name=topsavefile, camera=cameras.MYtopcam) #(camera=cameras.corcam)
        ccf_scenesag.close()
# =============================================================================
#     for actor in cell_actors:
#         ccf_scenesag.remove(actor)
# =============================================================================


