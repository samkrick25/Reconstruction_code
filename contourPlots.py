# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:54:33 2026
contour density plots
@author: samkr
"""

import pickle
from reconstructions.utils import preprocess_funcs
from reconstructions.utils.filedirs import frequenciespkl, somaspkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from brainrender import Scene
from brainrender.actors import Line, Points
import vedo

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'arial'

MIDLINE = 5700
def mirror_coords(dict):
    for soma, coords in dict.items():
        if coords['z'] > MIDLINE:
            diff = coords['z'] - MIDLINE
            coords['z'] = MIDLINE - diff

def get_coords(dict, refList, label=None):
    if label:
        return np.array([[coords['z'], coords['y'], coords['x'], label] for soma, coords in dict.items() if soma in refList])
    else:
        return np.array([[coords['x'], coords['y'], coords['z']] for soma, coords in dict.items() if soma in refList])

somas = pickle.load(open(somaspkl, 'rb'))
frequencies = pickle.load(open(frequenciespkl, 'rb')).T
merged = preprocess_funcs.merge_regions(frequencies)

motorNuclei = [[],[],[]]
for cell, targets in merged.iterrows():
    MNs = [targets['XII'], targets['V'], targets['VII']]
    if any(MNs):
        maxIndex = MNs.index(max(MNs))
        motorNuclei[maxIndex].append(cell)
        
for soma, coords in somas.items():
    mirror_coords({soma: coords})

XIIcoords = get_coords(somas, motorNuclei[0], label='XII')
Vcoords = get_coords(somas, motorNuclei[1], label='V')
VIIcoords = get_coords(somas, motorNuclei[2], label='VII')
data = np.concatenate([XIIcoords, Vcoords, VIIcoords], axis=0)

geyser = sns.load_dataset('geyser')

allcoords = pd.DataFrame(data, columns=['x', 'y', 'z', 'target'])
xydat = allcoords[['x', 'y', 'target']].astype({'x':'float', 'y':'float'})
xzdat = allcoords[['x', 'z', 'target']].astype({'x':'float', 'z':'float'})
zydat = allcoords[['z', 'y', 'target']].astype({'z':'float', 'y':'float'})

fig, [xyax, xzax, zyax] = plt.subplots(3, 1, figsize=(6, 14))
sns.kdeplot(xydat, x='x', y='y', ax=xyax, hue='target', levels=5)
sns.kdeplot(xzdat, x='x', y='z', ax=xzax, hue='target', levels=5)
sns.kdeplot(zydat, x='z', y='y', ax=zyax, hue='target', levels=5)

XIInolabel = get_coords(somas, motorNuclei[0])
Vnolabel = get_coords(somas, motorNuclei[1])
VIInolabel = get_coords(somas, motorNuclei[2])
points = np.concatenate([XIInolabel, Vnolabel, VIInolabel], axis=0)

ccf_scene=Scene(atlas_name='allen_mouse_10um')
horzplane1 = ccf_scene.atlas.get_plane((0,4000,0),plane='horizontal')
horzplane2 = ccf_scene.atlas.get_plane((0,4001,0),norm=(0,-1,0), plane='horizontal')
ccf_scene.slice(horzplane1)
ccf_scene.slice(horzplane2)

# ── Fix the third dimension using the mean of your data 
# (or substitute any specific coordinate value you prefer)
x_fixed = allcoords['x'].astype(float).mean()  # AP
y_fixed = allcoords['y'].astype(float).mean()  # DV
z_fixed = allcoords['z'].astype(float).mean()  # LR

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


# ── Each entry: (axes object, (seaborn x=, seaborn y=), fixed CCF axis, fixed value)
# Since coordinates are identical between the data and brainrender, no transform needed
plane_configs = [
    (xyax, ('x', 'y'), 'z', z_fixed),   # xy plot: fix z (LR)
    (xzax, ('x', 'z'), 'y', y_fixed),   # xz plot: fix y (DV)
    (zyax, ('z', 'y'), 'x', x_fixed),   # zy plot: fix x (AP)
]
chosen = plane_configs[1]

ax, free_axes, fixed_axis, fixed_val = chosen
for pts3d, rgba in extract_contours_3d(ax, free_axes, fixed_axis, fixed_val):
    actor = vedo.Line(pts3d, closed=True).lw(3).c(rgba[:3]).alpha(rgba[3])
    ccf_scene.add(actor)

plt.close('all')
ccf_scene.add_brain_region('root', alpha=0.05)
ccf_scene.render()
