# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:07:29 2026
show mossy fiber soma locations
@author: samkr
"""

from brainrender import Scene, settings
from brainrender.actors import Points
from reconstructions.utils.filedirs import somaspkl
from reconstructions.utils import cameras
import pickle
import numpy as np

MIDLINEZ = 5700

def mirror(node):
    z = node[2]
    diff = MIDLINEZ-z
    node[2] = MIDLINEZ+diff
    return node
    
mossys = ['AA0922', 'AA1263', 'N010-651324', 'N013-703070', 'N016-715345-HD', 'N017-703070', 'N017-715345-YV', 'N022-703070',
          'N023-715346-PC', 'N024-715345-SA', 'N030-651895', 'N031-651895', 'N031-715345-DS', 'N035-674191-FMR', 'N037-674185-IB',
          'N038-674185', 'N041-674191-AR', 'N044-674191-SP', 'N056-686955-JN', 'N057-686955-SA', 'N113-708369-JN', 
          'N114-708369-HS', 'N115-708369-BP']
somas = pickle.load(open(somaspkl, 'rb'))

settings.SHOW_AXES=None
settings.OFFSCREEN=True
settings.ROOT_ALPHA = 0
ccf_scene = Scene(atlas_name='allen_mouse_10um')
root = ccf_scene.get_actors()[0]
root._needs_silhouette=False

ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.2)
ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.2, color='pink')

medplane=ccf_scene.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
ccf_scene.slice(plane=medplane)

somacoords = np.array([mirror([soma['x'], soma['y'], soma['z']]) if soma['z'] < MIDLINEZ else [soma['x'], soma['y'], soma['z']] 
                       for cell, soma in somas.items() if cell in mossys])

somapoints = Points(somacoords, colors='black')
ccf_scene.add(somapoints)

savedir = r'images'
ccf_scene.render(camera=cameras.corcam)
# =============================================================================
# ccf_scene.screenshot(name=savedir+'\\'+'mossySomas_cor.png', camera=cameras.corcam, scale=3)
# ccf_scene.close()
# =============================================================================
