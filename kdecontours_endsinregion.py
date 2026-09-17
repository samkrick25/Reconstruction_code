# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:39:15 2026
kde contours for any region by phenotype
@author: samkr
"""

#%% imports
from reconstructions.utils import preprocess_funcs as pp
from reconstructions.utils.load_data import load_neurons, get_axonal_endpoints, get_nodes_in_region
from reconstructions.utils import filedirs
from reconstructions.utils.cellLists import cells_by_pheno

#%%
#set up paths and data structures
jsons = filedirs.allcoordswapped

ends_by_pheno = {}

pColors = {'sensory': 'firebrick', 'premotor': 'royalblue', 'mossy': 'seagreen', 'GRN': 'purple', 'sensorimotor': 'pink',
           'vestibular': 'navy', 'forebrain': 'orange'}

#%%
#create a mapping from cell name to pheno identity
def map_val_to_key(d):
    nd = {}
    for k, v in d.items():
        for val in v:
            nd[val]=k
    return nd

cells_to_pheno = map_val_to_key(cells_by_pheno)

#%% load cells
cells = load_neurons(jsons)

#%% get ends
ends_by_cell = get_axonal_endpoints(cells)
ends_by_cell = {cell: info['ends'] for cell, info in ends_by_cell.items()}

#%% get ends in PBN (or whtv your region of choice is)
PBends = get_nodes_in_region(ends_by_cell, regions='PB', ontlevel='structure', parcellated=False, mirror=True, kind='by_cell')

#%% get coordinates of each cell's ends to be put into brainrender
coords_by_cell = {cell: pp.get_coords(ends, dim='all') for cell, ends in PBends.items()}

#%% get coordinates for each phenotype
import numpy as np

points_by_pheno = {'sensory': np.empty((0,3)),
                   'premotor': np.empty((0,3)),
                   'sensorimotor': np.empty((0,3)),
                   'GRN': np.empty((0,3)),
                   'forebrain': np.empty((0,3)),
                   'vestibular': np.empty((0,3)),
                   'mossy': np.empty((0,3)),
                   }
for cell, coords in coords_by_cell.items():
    if coords.size > 0:
        pheno = cells_to_pheno[cell]
        points_by_pheno[pheno] = np.append(points_by_pheno[pheno], coords, axis=0)

#%% contour plotting fcn
def extract_contours_3d(ax, free_axes, fixed_axis, fixed_val):
    """
    Pull 2D contour paths from a seaborn kdeplot axes and
    reconstruct them as 3D points in brainrender/CCF space.

    Parameters
    ----------
    ax         : matplotlib Axes containing the kdeplot
    free_axes  : tuple of two strings, matching the seaborn x= and y= kwargs
                 e.g. ('x', 'y') meaning plot x-axis = CCF x, plot y-axis = CCF y
    fixed_axis : string, the CCF axis not shown in this plot ('x', 'y', or 'z')
    fixed_val  : float, the value to pin that fixed axis to

    Returns
    -------
    list of (pts_3d np.ndarray (N,3),  rgba np.ndarray (4,))
    """
    axis_index = {'x': 0, 'y': 1, 'z': 2}
    horiz, vert = free_axes           # which CCF axis is on plot x and plot y
    fixed = fixed_axis

    contours = []
    for collection in ax.collections:
        ec = collection.get_edgecolors()
        fc = collection.get_facecolors()
        color = ec[0] if len(ec) else (fc[0] if len(fc) else np.array([0.5, 0.5, 0.5, 1.0]))

        for path in collection.get_paths():
            verts = path.vertices      # shape (N, 2)
            if len(verts) < 4:
                continue
            n = len(verts)

            # Build the 3D array by placing each plot axis back to its CCF axis
            pts3d = np.empty((n, 3))
            pts3d[:, axis_index[horiz]] = verts[:, 0]   # plot x-axis → CCF axis
            pts3d[:, axis_index[vert]]  = verts[:, 1]   # plot y-axis → CCF axis
            pts3d[:, axis_index[fixed]] = fixed_val      # third axis → fixed value

            contours.append((pts3d, color))

    return contours

#%% set up brainrender scene
from brainrender import settings, Scene
from reconstructions.utils import cameras as c

settings.INTERACTIVE = False
settings.OFFSCREEN = True
settings.BACKGROUND_COLOR = 'black'
settings.ROOT_ALPHA = 0.075
settings.ROOT_COLOR='black'


scene = Scene(atlas_name = 'allen_mouse_10um')

scene.get_actors()[0]._needs_silhouette = False

#%% define function to manipulate coordinates for each phenotype into what needs
#to be passed into contour creating fcn
def get_kde_actors(coordsdict, dims):
    ...