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

celldir = r'C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\premotorsU19\N034-674191-IB'

print('setting scene')
ccf_scene = Scene(atlas_name='allen_mouse_10um')
root = ccf_scene.get_actors()[0]
root._needs_silhouette = False
settings.SHOW_AXES = False
print('scene set')
print('adding brain regions')
ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.2)
ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.2, color='pink')
ccf_scene.add_brain_region('XII', silhouette=False, color='purple', alpha=0.2)
ccf_scene.add_brain_region('V', silhouette=False, color='green', alpha=0.2)
ccf_scene.add_brain_region('VII', silhouette=False, color='blue', alpha=0.2)

colorsone = ['green', 'blue', 'purple']
colorsmix = ['orangered', 'red', 'black']

for file in tqdm(os.listdir(celldir), desc='Loading neurons'):
    filestr = file.split('.')[0]
    cellname = filestr
    filename = os.path.join(celldir, file)
    lines = pf.swap_for_brainrender(filename, axon=colorsmix[0], skip_dendrite=True)
    for actor in tqdm(lines, desc='Adding actors'):
        ccf_scene.add(actor)
            
ccf_scene.screenshot(name=celldir+'\\'+cellname+'top.png', camera=cameras.MYtopcam, scale=3)
ccf_scene.close()