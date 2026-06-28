# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:28:17 2026

@author: samkr
"""

from brainrender import Scene
from brainrender import settings
from reconstructions.utils import preprocess_funcs as pf
from reconstructions.utils import cameras
import numpy as np

#path to swc to be visualized
neuron = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swcsfromjson\AA0503.swc"

settings.SHOW_AXES=None
settings.INTERACTIVE=True
settings.OFFSCREEN=False

#set up brainrender scene
ccf_scene = Scene(atlas_name='allen_mouse_10um', root=True)
root = ccf_scene.get_actors()[0]
root._needs_silhouette = False
settings.ROOT_COLOR=[0.8, 0.8, 0.8]
settings.ROOT_ALPHA=0.2
print('scene set')

regions = ['tsp']
copies = [ccf_scene.atlas.get_region(r).mesh.clone() for r in regions]
cut = pf.get_mesh_onehem(ccf_scene, mesh=copies, hem='left')
verts = [m.vertices for m in cut]

#add desired brain regions
ccf_scene.add_brain_region('IRN', silhouette=False, color='pink', alpha=0.2)
ccf_scene.add_brain_region('PARN', silhouette=False, alpha=0.2, color='pink')
ccf_scene.add_brain_region('PVH', silhouette=False, color='orange', alpha=0.2)
ccf_scene.add_brain_region('tsp', silhouette=False, color='purple', alpha=0.2)
# =============================================================================
# ccf_scene.add_brain_region('V', silhouette=False, color='yellow', alpha=0.2)
# ccf_scene.add_brain_region('SPVO', silhouette=False, color='cyan', alpha=0.2)
# ccf_scene.add_brain_region('SPVI', silhouette=False, color='red', alpha=0.2)
# =============================================================================

#add neuron, first create line actors
lines = pf.swap_for_brainrender(neuron, skip_dendrite=True, neurite_radius=10, soma_radius=25, mesh=verts)
for actor in lines:
    ccf_scene.add(actor)
    
ccf_scene.render(camera=cameras.MYtopcam)

'''
    i     print info about the last clicked object     
    I     print color of the pixel under the mouse     
    Y     show the pipeline for this object as a graph 
    <- -> use arrows to reduce/increase opacity        
    x     toggle mesh visibility                       
    w     toggle wireframe/surface style               
    l     toggle surface edges visibility              
    p/P   hide surface faces and show only points      
    1-3   cycle surface color (2=light, 3=dark)        
    4     cycle color map (press shift-4 to go back)   
    5-6   cycle point-cell arrays (shift to go back)   
    7-8   cycle background and gradient color          
    09+-  cycle axes styles (on keypad, or press +/-)  
    k     cycle available lighting styles              
    K     toggle shading as flat or phong              
    A     toggle anti-aliasing                         
    D     toggle depth-peeling (for transparencies)    
    U     toggle perspective/parallel projection       
    o/O   toggle extra light to scene and rotate it    
    a     toggle interaction to Actor Mode             
    n     toggle surface normals                       
    r     reset camera position                        
    R     reset camera to the closest orthogonal view  
    .     fly camera to the last clicked point         
    C     print the current camera parameters state    
    X     invoke a cutter widget tool                  
    S     save a screenshot of the current scene       
    E/F   export 3D scene to numpy file or X3D         
    q     return control to python script              
    Esc   abort execution and exit python kernel  
'''