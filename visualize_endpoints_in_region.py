# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:24:51 2026
visualize endpoints of IRN/PARN cells in any region(s)
@author: economolab
"""

from reconstructions.utils import visualize, cameras, plotting_funcs
from brainrender import Scene
import numpy as np

antnodes, postnodes = visualize.show_endpoints('NTS', root=False, colors=['orange'], alphas=[0.2], unilat=True, camera=cameras.sagcam)

antnonempty=[]
for cell, nodes in antnodes.items():
    if nodes:
        antnonempty.append(nodes)
antflat = [node for cell in antnonempty for node in cell]

postnonempty=[]
for cell, nodes in postnodes.items():
    if nodes:
        postnonempty.append(nodes)
postflat = [node for cell in postnonempty for node in cell]

ccf_v3 = Scene(atlas_name='allen_mouse_10um', root=False)
ccf_v3.add_brain_region('NTS', hemisphere='left')
medplane=ccf_v3.atlas.get_plane(plane='sagittal',norm=(0,0,-1))
ccf_v3.slice(plane=medplane)
actors = ccf_v3.get_actors()
regmesh = actors[0]
regmeshverts = regmesh.mesh.vertices
regap = [v[0] for v in regmeshverts]
regdv = [v[1] for v in regmeshverts]
regml = [v[2] for v in regmeshverts]


fig, axes = plotting_funcs.comp_node_dist(antflat, postflat, labels=['anterior IRN/PARN', 'posterior IRN/PARN'], suptitle = 'Distribution of IRN/PARN endpoints in NTS')

axes[0].axvline(x=np.max(regap))
axes[0].axvline(x=np.min(regap))
axes[1].axvline(x=np.max(regdv))
axes[1].axvline(x=np.min(regdv))
axes[2].axvline(x=np.max(regml))
#kinda have to estimate this rn cuz i don't have time to find exact medial bound
axes[2].axvline(x=5700)
fig.show()