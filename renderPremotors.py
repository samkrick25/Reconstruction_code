# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:28:33 2026
render premotor cells for U19
@author: samkr
"""

import os
from brainrender import Scene
from brainrender import settings
from reconstructions.utils import cameras
from tqdm import tqdm
from reconstructions.utils import preprocess_funcs as pf

celldir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\premotorsU19\N071-709222'
savedir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\premotorsU19'
#celldir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\premotorsU19\alltorender'
print('setting scene')

ccf_scene = Scene(atlas_name='allen_mouse_10um')
root = ccf_scene.get_actors()[0]
root._silhouette_kwargs['lw'] = 20

horzplane1 = ccf_scene.atlas.get_plane((0,4000,0),plane='horizontal')
horzplane2 = ccf_scene.atlas.get_plane((0,4001,0),norm=(0,-1,0), plane='horizontal')
ccf_scene.slice(horzplane1)
ccf_scene.slice(horzplane2)
#root.make_silhouette({'lw':10})
# =============================================================================
# root._needs_silhouette = True
# settings.SHOW_AXES = False
# settings.ROOT_COLOR='white'
# =============================================================================
print('scene set')
# =============================================================================
# ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.2)
# ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.2, color='pink')
# ccf_scene.add_brain_region('XII', silhouette=False, color='purple', alpha=0.2)
# ccf_scene.add_brain_region('V', silhouette=False, color='green', alpha=0.2)
# ccf_scene.add_brain_region('VII', silhouette=False, color='blue', alpha=0.2)
# =============================================================================
rootcam = dict(
    pos=(6861.42, -108506, -5802.38),
    focal_point=(7080.29, 4292.90, -5022.47),
    viewup=(-1.00000, 0, 0),
    roll=74.3246,
    distance=112801,
    clipping_range=(103739, 123466),
)

# =============================================================================
# colors = ['green', 'red', 'blue', 'brown', 'black', 'purple']
# for i, file in enumerate(os.listdir(celldir)):
#     filestr = file.split('.')[0]
#     cellname = filestr
#     filename = os.path.join(celldir, file)
#     lines = pf.swap_for_brainrender(filename, axon=colors[i], skip_dendrite=True)
#     for line in lines:
#         ccf_scene.add(line)
# =============================================================================
ccf_scene.screenshot(name=savedir+'\\'+'orienting3.png', camera=rootcam, scale=3)
ccf_scene.close()
#ccf_scene.render(camera=cameras.MYtopcam)
print('adding brain regions')
# =============================================================================
# ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.2)
# ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.2, color='pink')
# ccf_scene.add_brain_region('XII', silhouette=False, color='purple', alpha=0.2)
# ccf_scene.add_brain_region('V', silhouette=False, color='green', alpha=0.2)
# ccf_scene.add_brain_region('VII', silhouette=False, color='blue', alpha=0.2)
# 
# colorsone = ['green', 'blue', 'purple']
# colorsmix = ['red', 'brown', 'black']
# 
# for file in tqdm(os.listdir(celldir), desc='Loading neurons'):
#     filestr = file.split('.')[0]
#     cellname = filestr
#     filename = os.path.join(celldir, file)
#     lines = pf.swap_for_brainrender(filename, axon=colorsone[2], skip_dendrite=True, lw=5)
#     for actor in tqdm(lines, desc='Adding actors'):
#         ccf_scene.add(actor)
# 
# =============================================================================

# =============================================================================
# ccf_scene.screenshot(name=celldir+'\\'+cellname+'top.svg', camera=cameras.MYtopcam, scale=3)
# ccf_scene.close()
# =============================================================================

